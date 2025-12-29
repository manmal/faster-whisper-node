//! faster-whisper-node engine
//!
//! High-performance Whisper transcription for Node.js via whisper.cpp with Metal acceleration.

mod audio;
mod download;
mod streaming;
mod vad;
mod word_timestamps;
#[cfg(feature = "silero-vad")]
mod silero_vad;

use napi_derive::napi;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

use vad::{EnergyVad, VadOptions as InternalVadOptions};
#[cfg(feature = "silero-vad")]
use silero_vad::{SileroVad, SileroVadConfig};
use word_timestamps::parse_timestamped_text;

/// Check if Metal (GPU acceleration) is available on macOS
fn is_metal_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        true // Metal is always available on macOS with Apple Silicon or recent Intel Macs
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Check if CUDA is available (for Linux/Windows with NVIDIA GPU)
fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // whisper-rs handles CUDA detection internally
        true
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

// Re-export download functions
pub use download::{
    default_cache_dir, 
    model_path_for_size, 
    is_model_downloaded,
    available_model_sizes,
};

/// Word with timing information (for word-level timestamps)
#[napi(object)]
#[derive(Clone, Debug)]
pub struct Word {
    /// The word text
    pub word: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Word probability/confidence (0.0 to 1.0)
    pub probability: f64,
}

/// Transcription segment with timing and confidence information
#[napi(object)]
#[derive(Clone, Debug)]
pub struct Segment {
    /// Segment ID (0-indexed)
    pub id: u32,
    /// Seek position in audio frames
    pub seek: u32,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Transcribed text
    pub text: String,
    /// Token IDs
    pub tokens: Vec<u32>,
    /// Decoding temperature used
    pub temperature: f64,
    /// Average log probability
    pub avg_logprob: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Probability of no speech
    pub no_speech_prob: f64,
    /// Word-level timestamps (if wordTimestamps option was enabled)
    pub words: Option<Vec<Word>>,
}

/// Voice Activity Detection (VAD) options
#[napi(object)]
#[derive(Clone, Debug)]
pub struct VadOptions {
    /// Speech detection threshold (0.0 to 1.0, default: 0.5)
    pub threshold: Option<f64>,
    /// Minimum speech duration in milliseconds (default: 250)
    pub min_speech_duration_ms: Option<u32>,
    /// Maximum speech duration in seconds (default: 30)
    pub max_speech_duration_s: Option<f64>,
    /// Minimum silence duration in milliseconds to split segments (default: 2000)
    pub min_silence_duration_ms: Option<u32>,
    /// Analysis window size in milliseconds (default: 30)
    pub window_size_ms: Option<u32>,
    /// Padding around speech segments in milliseconds (default: 400)
    pub speech_pad_ms: Option<u32>,
    /// Use Silero VAD (neural network) instead of energy-based detection.
    /// Requires the silero-vad feature and model file. More accurate but slower.
    pub use_silero: Option<bool>,
    /// Path to Silero VAD ONNX model file (optional, will use default cache location)
    pub silero_model_path: Option<String>,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: None,
            min_speech_duration_ms: None,
            max_speech_duration_s: None,
            min_silence_duration_ms: None,
            window_size_ms: None,
            speech_pad_ms: None,
            use_silero: None,
            silero_model_path: None,
        }
    }
}

/// Transcription options
#[napi(object)]
#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    /// Source language (e.g., "en", "de", "fr"). If not set, language is auto-detected.
    pub language: Option<String>,
    /// Task to perform: "transcribe" or "translate"
    pub task: Option<String>,
    /// Beam size for beam search (default: 5, set to 1 for greedy search)
    pub beam_size: Option<u32>,
    /// Beam search patience factor (default: 1.0)
    pub patience: Option<f64>,
    /// Exponential penalty applied to length during beam search (default: 1.0)
    pub length_penalty: Option<f64>,
    /// Penalty for repetition (default: 1.0, set > 1 to penalize)
    pub repetition_penalty: Option<f64>,
    /// Prevent repetitions of ngrams with this size (default: 0, disabled)
    pub no_repeat_ngram_size: Option<u32>,
    /// Sampling temperature (default: 1.0)
    pub temperature: Option<f64>,
    /// Suppress blank outputs at beginning (default: true)
    pub suppress_blank: Option<bool>,
    /// Maximum generation length (default: 448)
    pub max_length: Option<u32>,
    /// Include word-level timestamps (default: false)
    pub word_timestamps: Option<bool>,
    /// Initial prompt to provide context
    pub initial_prompt: Option<String>,
    /// Prefix for the first segment
    pub prefix: Option<String>,
    /// Suppress tokens (comma-separated IDs or special tokens)
    pub suppress_tokens: Option<String>,
    /// Apply condition on previous text (default: true)
    pub condition_on_previous_text: Option<bool>,
    /// Compression ratio threshold for detecting failed decodings
    pub compression_ratio_threshold: Option<f64>,
    /// Log probability threshold for detecting failed decodings
    pub log_prob_threshold: Option<f64>,
    /// No speech probability threshold
    pub no_speech_threshold: Option<f64>,
    /// Enable Voice Activity Detection to filter out silent portions (default: false)
    pub vad_filter: Option<bool>,
    /// VAD-specific options
    pub vad_options: Option<VadOptions>,
    /// Hallucination silence threshold in seconds (default: None)
    /// If a segment's duration per word exceeds this, it's likely a hallucination
    pub hallucination_silence_threshold: Option<f64>,
    /// Number of parallel processors for transcription (default: 1)
    /// Using more processors can speed up long audio files but may have 
    /// reduced accuracy at chunk boundaries.
    /// Note: Word timestamps are not supported when n_processors > 1.
    pub n_processors: Option<u32>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            language: None,
            task: None,
            beam_size: None,
            patience: None,
            length_penalty: None,
            repetition_penalty: None,
            no_repeat_ngram_size: None,
            temperature: None,
            suppress_blank: None,
            max_length: None,
            word_timestamps: None,
            initial_prompt: None,
            prefix: None,
            suppress_tokens: None,
            condition_on_previous_text: None,
            compression_ratio_threshold: None,
            log_prob_threshold: None,
            no_speech_threshold: None,
            vad_filter: None,
            vad_options: None,
            hallucination_silence_threshold: None,
            n_processors: None,
        }
    }
}

/// Options for model loading
#[napi(object)]
#[derive(Clone, Debug)]
pub struct ModelOptions {
    /// Device to use: "cpu", "cuda", "metal", or "auto" (default: "auto")
    pub device: Option<String>,
    /// Compute type (not used in whisper-rs, kept for API compatibility)
    pub compute_type: Option<String>,
    /// Number of CPU threads (0 = auto)
    pub cpu_threads: Option<u32>,
    /// GPU device index (for multi-GPU systems)
    pub device_index: Option<u32>,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            device: None,
            compute_type: None,
            cpu_threads: None,
            device_index: None,
        }
    }
}

/// Transcription result with full segment data
#[napi(object)]
#[derive(Clone, Debug)]
pub struct TranscriptionResult {
    /// All transcription segments
    pub segments: Vec<Segment>,
    /// Detected or specified language
    pub language: String,
    /// Language detection probability
    pub language_probability: f64,
    /// Total audio duration in seconds
    pub duration: f64,
    /// Duration after VAD filtering (if enabled)
    pub duration_after_vad: f64,
    /// Full transcription text
    pub text: String,
}

/// Language detection result
#[napi(object)]
#[derive(Clone, Debug)]
pub struct LanguageDetectionResult {
    /// Detected language code
    pub language: String,
    /// Detection probability
    pub probability: f64,
}

/// Batch transcription result for a single file
#[napi(object)]
#[derive(Clone, Debug)]
pub struct BatchTranscriptionItem {
    /// Original file path
    pub file_path: String,
    /// Transcription result (None if error)
    pub result: Option<TranscriptionResult>,
    /// Error message (None if success)
    pub error: Option<String>,
    /// Current file index
    pub current_index: u32,
}

/// Sample rate for Whisper (16kHz)
const WHISPER_SAMPLE_RATE: u32 = 16000;

#[napi]
pub struct Engine {
    ctx: WhisperContext,
    num_threads: u32,
}

#[napi]
impl Engine {
    /// Create a new transcription engine from a model path or size
    /// 
    /// # Arguments
    /// * `model_path` - Either a path to a GGML model file, or a model size 
    ///                  alias ("tiny", "base", "small", "medium", "large-v2", "large-v3")
    #[napi(constructor)]
    pub fn new(model_path: String) -> napi::Result<Self> {
        Self::with_options(model_path, None)
    }

    /// Create a new transcription engine with options
    #[napi(factory)]
    pub fn with_options(model_path: String, options: Option<ModelOptions>) -> napi::Result<Self> {
        let opts = options.unwrap_or_default();
        
        // Resolve path (could be alias like "tiny" or actual path)
        let resolved_path = download::resolve_model_path(&model_path);
        
        // Check if model exists
        if !std::path::Path::new(&resolved_path).exists() {
            // Check if it's a known alias that needs downloading
            if download::get_repo_for_size(&model_path).is_some() {
                return Err(napi::Error::from_reason(format!(
                    "Model '{}' not found. Download it first using: await downloadModel('{}')",
                    model_path, model_path
                )));
            }
            return Err(napi::Error::from_reason(format!(
                "Model not found at: {}", resolved_path
            )));
        }
        
        // Create WhisperContext with parameters
        let mut ctx_params = WhisperContextParameters::default();
        
        // Enable GPU acceleration based on device option
        let use_gpu = match opts.device.as_deref() {
            Some("cpu") | Some("CPU") => false,
            Some("metal") | Some("Metal") | Some("METAL") => true,
            Some("cuda") | Some("CUDA") => true,
            Some("auto") | Some("AUTO") | None => {
                is_metal_available() || is_cuda_available()
            }
            _ => false,
        };
        ctx_params.use_gpu(use_gpu);
        
        let ctx = WhisperContext::new_with_params_and_state(&resolved_path, ctx_params)
            .map_err(|e| napi::Error::from_reason(format!("Failed to load model: {}", e)))?;
        
        let num_threads = opts.cpu_threads.unwrap_or(0);
        
        Ok(Self { ctx, num_threads })
    }

    /// Transcribe audio file (supports WAV, MP3, FLAC, OGG, M4A)
    #[napi]
    pub fn transcribe_file(
        &self,
        audio_path: String,
        options: Option<TranscribeOptions>,
    ) -> napi::Result<TranscriptionResult> {
        let opts = options.unwrap_or_default();
        
        let samples = audio::decode_audio_file(&audio_path)
            .map_err(|e| napi::Error::from_reason(format!("Failed to decode audio: {}", e)))?;
        
        self.transcribe_samples_internal(&samples, &opts)
    }

    /// Legacy: transcribe from WAV file path, returns structured segments
    #[napi]
    pub fn transcribe_segments(
        &self,
        audio_path: String,
        options: Option<TranscribeOptions>,
    ) -> napi::Result<TranscriptionResult> {
        self.transcribe_file(audio_path, options)
    }

    /// Simple transcription returning just the text (backward compatible)
    #[napi]
    pub fn transcribe(&self, audio_file: String) -> napi::Result<String> {
        let result = self.transcribe_file(audio_file, None)?;
        Ok(result.text)
    }

    /// Transcribe with options, returning just the text
    #[napi]
    pub fn transcribe_with_options(
        &self,
        audio_file: String,
        options: TranscribeOptions,
    ) -> napi::Result<String> {
        let result = self.transcribe_file(audio_file, Some(options))?;
        Ok(result.text)
    }

    /// Transcribe from a Buffer containing audio data (any supported format)
    #[napi]
    pub fn transcribe_buffer(
        &self,
        buffer: napi::bindgen_prelude::Buffer,
        options: Option<TranscribeOptions>,
    ) -> napi::Result<TranscriptionResult> {
        let opts = options.unwrap_or_default();
        
        let samples = audio::decode_audio_buffer(&buffer)
            .map_err(|e| napi::Error::from_reason(format!("Failed to decode audio buffer: {}", e)))?;
        
        self.transcribe_samples_internal(&samples, &opts)
    }

    /// Transcribe from raw Float32Array samples (must be 16kHz mono, normalized to [-1, 1])
    #[napi]
    pub fn transcribe_samples(
        &self,
        samples: Vec<f64>,
        options: Option<TranscribeOptions>,
    ) -> napi::Result<TranscriptionResult> {
        let opts = options.unwrap_or_default();
        // Convert f64 to f32
        let samples_f32: Vec<f32> = samples.iter().map(|&x| x as f32).collect();
        
        self.transcribe_samples_internal(&samples_f32, &opts)
    }

    /// Detect the language of audio
    /// Note: This performs a quick transcription to detect language.
    /// For efficiency, only the first 30 seconds are analyzed.
    #[napi]
    pub fn detect_language(
        &self,
        audio_path: String,
    ) -> napi::Result<LanguageDetectionResult> {
        let samples = audio::decode_audio_file(&audio_path)
            .map_err(|e| napi::Error::from_reason(format!("Failed to decode audio: {}", e)))?;
        
        // Use first 30 seconds max
        let max_samples = (30.0 * WHISPER_SAMPLE_RATE as f64) as usize;
        let detection_samples: &[f32] = if samples.len() > max_samples {
            &samples[..max_samples]
        } else {
            &samples
        };
        
        // Do a quick transcription with language=None to trigger auto-detection
        let opts = TranscribeOptions::default();
        let result = self.transcribe_samples_internal(detection_samples, &opts)?;
        
        Ok(LanguageDetectionResult {
            language: result.language,
            probability: result.language_probability,
        })
    }

    /// Detect language from buffer
    #[napi]
    pub fn detect_language_buffer(
        &self,
        buffer: napi::bindgen_prelude::Buffer,
    ) -> napi::Result<LanguageDetectionResult> {
        let samples = audio::decode_audio_buffer(&buffer)
            .map_err(|e| napi::Error::from_reason(format!("Failed to decode audio buffer: {}", e)))?;
        
        let max_samples = (30.0 * WHISPER_SAMPLE_RATE as f64) as usize;
        let detection_samples: &[f32] = if samples.len() > max_samples {
            &samples[..max_samples]
        } else {
            &samples
        };
        
        let opts = TranscribeOptions::default();
        let result = self.transcribe_samples_internal(detection_samples, &opts)?;
        
        Ok(LanguageDetectionResult {
            language: result.language,
            probability: result.language_probability,
        })
    }

    /// Get the expected sampling rate (16000 Hz for Whisper)
    #[napi]
    pub fn sampling_rate(&self) -> u32 {
        WHISPER_SAMPLE_RATE
    }

    /// Check if the model is multilingual
    #[napi]
    pub fn is_multilingual(&self) -> bool {
        self.ctx.is_multilingual()
    }

    /// Get the number of supported languages
    #[napi]
    pub fn num_languages(&self) -> u32 {
        // whisper.cpp supports ~100 languages
        99
    }

    // Internal transcription implementation
    fn transcribe_samples_internal(
        &self,
        samples: &[f32],
        opts: &TranscribeOptions,
    ) -> napi::Result<TranscriptionResult> {
        // Calculate original duration
        let duration = samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;
        
        // Apply VAD filtering if enabled
        let (processed_samples, vad_offset_map) = if opts.vad_filter.unwrap_or(false) {
            let vad_options = opts.vad_options.as_ref();
            
            // Check if Silero VAD is requested and available
            #[cfg(feature = "silero-vad")]
            let use_silero = vad_options.and_then(|v| v.use_silero).unwrap_or(false);
            #[cfg(not(feature = "silero-vad"))]
            let use_silero = false;
            
            if use_silero {
                #[cfg(feature = "silero-vad")]
                {
                    let model_path = vad_options
                        .and_then(|v| v.silero_model_path.clone())
                        .unwrap_or_else(|| {
                            download::default_cache_dir()
                                .join("silero_vad.onnx")
                                .to_string_lossy()
                                .into_owned()
                        });
                    
                    let config = SileroVadConfig {
                        threshold: vad_options.and_then(|v| v.threshold).unwrap_or(0.5) as f32,
                        min_speech_duration_ms: vad_options.and_then(|v| v.min_speech_duration_ms).unwrap_or(250),
                        min_silence_duration_ms: vad_options.and_then(|v| v.min_silence_duration_ms).unwrap_or(100),
                        speech_pad_ms: vad_options.and_then(|v| v.speech_pad_ms).unwrap_or(30),
                    };
                    
                    let mut silero_vad = SileroVad::new(&model_path, config)
                        .map_err(|e| napi::Error::from_reason(format!("Failed to load Silero VAD: {}", e)))?;
                    
                    silero_vad.filter_audio(samples)
                        .map_err(|e| napi::Error::from_reason(format!("Silero VAD failed: {}", e)))?
                }
                #[cfg(not(feature = "silero-vad"))]
                {
                    return Err(napi::Error::from_reason(
                        "Silero VAD requested but silero-vad feature not enabled".to_string()
                    ));
                }
            } else {
                // Use energy-based VAD
                let vad_opts = self.build_vad_options(vad_options);
                let vad = EnergyVad::new(WHISPER_SAMPLE_RATE, vad_opts);
                vad.filter_audio(samples)
            }
        } else {
            (samples.to_vec(), vec![(0.0, 0.0)])
        };
        
        let duration_after_vad = processed_samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;
        
        // Create state for this transcription
        let mut state = self.ctx.create_state()
            .map_err(|e| napi::Error::from_reason(format!("Failed to create state: {}", e)))?;
        
        // Build parameters inline to avoid lifetime issues
        let strategy = if opts.beam_size.unwrap_or(5) <= 1 {
            SamplingStrategy::Greedy { best_of: 1 }
        } else {
            SamplingStrategy::BeamSearch { 
                beam_size: opts.beam_size.unwrap_or(5) as i32,
                patience: opts.patience.unwrap_or(1.0) as f32,
            }
        };
        
        let mut params = FullParams::new(strategy);
        
        // Set number of threads
        let n_threads = if self.num_threads > 0 { self.num_threads as i32 } else { 4 };
        params.set_n_threads(n_threads);
        
        // Set language
        if let Some(ref lang) = opts.language {
            params.set_language(Some(lang));
        }
        
        // Set task (translate vs transcribe)
        if let Some(ref task) = opts.task {
            params.set_translate(task == "translate");
        }
        
        // Suppress non-speech
        if opts.suppress_blank.unwrap_or(true) {
            params.set_suppress_blank(true);
        }
        
        // Disable printing to stdout
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        
        // Initial prompt
        if let Some(ref prompt) = opts.initial_prompt {
            params.set_initial_prompt(prompt);
        }
        
        // Temperature
        if let Some(temp) = opts.temperature {
            params.set_temperature(temp as f32);
        }
        
        // No speech threshold
        if let Some(threshold) = opts.no_speech_threshold {
            params.set_no_speech_thold(threshold as f32);
        }
        
        // Determine if we want word timestamps
        let want_word_timestamps = opts.word_timestamps.unwrap_or(false);
        params.set_token_timestamps(want_word_timestamps);
        
        // Check if using parallel processing
        let n_processors = opts.n_processors.unwrap_or(1);
        let use_parallel = n_processors > 1;
        
        // Run transcription (parallel or sequential)
        if use_parallel {
            self.ctx.full_parallel(params, &processed_samples, n_processors as i32)
                .map_err(|e| napi::Error::from_reason(format!("Parallel transcription failed: {}", e)))?;
        } else {
            state.full(params, &processed_samples)
                .map_err(|e| napi::Error::from_reason(format!("Transcription failed: {}", e)))?;
        }
        
        // Build segments from results
        let mut segments = Vec::new();
        let mut full_text = String::new();
        let use_vad = opts.vad_filter.unwrap_or(false);
        let hallucination_threshold = opts.hallucination_silence_threshold;
        
        // Get segment count from context (parallel) or state (sequential)
        let num_segments = if use_parallel {
            self.ctx.full_n_segments()
        } else {
            state.full_n_segments()
        };
        
        for i in 0..num_segments {
            // Get segment data - parallel mode uses context methods, sequential uses state segment API
            let (raw_text, start_ts, end_ts, segment_for_words) = if use_parallel {
                let text = self.ctx.full_get_segment_text(i).unwrap_or_default();
                let start = self.ctx.full_get_segment_t0(i);
                let end = self.ctx.full_get_segment_t1(i);
                (text, start, end, None)
            } else {
                let segment = match state.get_segment(i) {
                    Some(s) => s,
                    None => continue,
                };
                let text = segment.to_str_lossy().map(|t| t.into_owned()).unwrap_or_default();
                let start = segment.start_timestamp();
                let end = segment.end_timestamp();
                (text, start, end, Some(segment))
            };
            
            let raw_text = raw_text.trim();
            if raw_text.is_empty() {
                continue;
            }
            
            // Convert from centiseconds to seconds
            let filtered_segment_start = start_ts as f64 / 100.0;
            let filtered_segment_end = end_ts as f64 / 100.0;
            
            // Convert to original audio time if VAD was used
            let (segment_start, segment_end) = if use_vad {
                (
                    vad::restore_timestamp(filtered_segment_start, &vad_offset_map),
                    vad::restore_timestamp(filtered_segment_end, &vad_offset_map),
                )
            } else {
                (filtered_segment_start, filtered_segment_end)
            };
            
            // Parse word-level timestamps if enabled
            let words = if want_word_timestamps {
                let num_tokens = if use_parallel {
                    self.ctx.full_n_tokens(i)
                } else {
                    segment_for_words.as_ref().map(|s| s.n_tokens()).unwrap_or(0)
                };
                
                let mut words_vec = Vec::new();
                for j in 0..num_tokens {
                    // Get token text and data from context (parallel) or state segment (sequential)
                    let (token_text, token_data, token_prob) = if use_parallel {
                        let text = self.ctx.full_get_token_text_lossy(i, j).unwrap_or_default();
                        let data = self.ctx.full_get_token_data(i, j);
                        let prob = self.ctx.full_get_token_prob(i, j);
                        (text, data, prob)
                    } else {
                        let segment = segment_for_words.as_ref().unwrap();
                        let token = match segment.get_token(j) {
                            Some(t) => t,
                            None => continue,
                        };
                        let text = token.to_str_lossy().map(|t| t.into_owned()).unwrap_or_default();
                        let data = token.token_data();
                        let prob = token.token_probability();
                        (text, data, prob)
                    };
                    
                    // Skip special tokens
                    if token_text.starts_with('[') || token_text.starts_with('<') {
                        continue;
                    }
                    
                    let word_start = token_data.t0 as f64 / 100.0;
                    let word_end = token_data.t1 as f64 / 100.0;
                    
                    let (adj_start, adj_end) = if use_vad {
                        (
                            vad::restore_timestamp(word_start, &vad_offset_map),
                            vad::restore_timestamp(word_end, &vad_offset_map),
                        )
                    } else {
                        (segment_start + word_start, segment_start + word_end)
                    };
                    
                    words_vec.push(Word {
                        word: token_text.trim().to_string(),
                        start: adj_start,
                        end: adj_end,
                        probability: token_prob as f64,
                    });
                }
                Some(words_vec)
            } else {
                None
            };
            
            // Hallucination detection
            if let Some(threshold) = hallucination_threshold {
                let segment_duration = segment_end - segment_start;
                let text_len = raw_text.split_whitespace().count();
                
                if text_len > 0 {
                    let duration_per_word = segment_duration / text_len as f64;
                    if duration_per_word > threshold {
                        continue; // Skip this segment as likely hallucination
                    }
                }
            }
            
            if !full_text.is_empty() {
                full_text.push(' ');
            }
            full_text.push_str(raw_text);
            
            segments.push(Segment {
                id: i as u32,
                seek: 0,
                start: segment_start,
                end: segment_end,
                text: raw_text.to_string(),
                tokens: vec![],
                temperature: opts.temperature.unwrap_or(0.0),
                avg_logprob: 0.0,
                compression_ratio: 0.0,
                no_speech_prob: 0.0,
                words,
            });
        }
        
        // Get detected language if available
        let detected_language = opts.language.clone().unwrap_or_else(|| "en".to_string());
        
        Ok(TranscriptionResult {
            segments,
            language: detected_language,
            language_probability: 0.0,
            duration,
            duration_after_vad,
            text: full_text,
        })
    }
    
    // Helper: build VadOptions from JS VadOptions
    fn build_vad_options(&self, opts: Option<&VadOptions>) -> InternalVadOptions {
        let mut vad_opts = InternalVadOptions::default();
        
        if let Some(o) = opts {
            if let Some(t) = o.threshold {
                vad_opts.threshold = t as f32;
            }
            if let Some(v) = o.min_speech_duration_ms {
                vad_opts.min_speech_duration_ms = v;
            }
            if let Some(v) = o.max_speech_duration_s {
                vad_opts.max_speech_duration_s = v as f32;
            }
            if let Some(v) = o.min_silence_duration_ms {
                vad_opts.min_silence_duration_ms = v;
            }
            if let Some(v) = o.window_size_ms {
                vad_opts.window_size_ms = v;
            }
            if let Some(v) = o.speech_pad_ms {
                vad_opts.speech_pad_ms = v;
            }
        }
        
        vad_opts
    }

}

// ============== Standalone Functions ==============

/// Get list of supported model size aliases
#[napi]
pub fn available_models() -> Vec<String> {
    download::available_model_sizes().iter()
        .map(|s| s.to_string())
        .collect()
}

/// Check if a model is downloaded
#[napi]
pub fn is_model_available(size: String) -> bool {
    download::is_model_downloaded(&size, None)
}

/// Get the path where a model would be stored
#[napi]
pub fn get_model_path(size: String) -> String {
    download::model_path_for_size(&size, None)
        .to_string_lossy()
        .into_owned()
}

/// Get the default cache directory for models
#[napi]
pub fn get_cache_dir() -> String {
    download::default_cache_dir()
        .to_string_lossy()
        .into_owned()
}

/// Download a model (async)
/// Returns the path to the downloaded model
#[napi]
pub async fn download_model(
    size: String,
    cache_dir: Option<String>,
) -> napi::Result<String> {
    let cache_path = cache_dir.map(std::path::PathBuf::from);
    
    let result = download::download_model(
        &size,
        cache_path.as_deref(),
        None::<fn(f64)>,
    ).await
        .map_err(|e| napi::Error::from_reason(format!("Download failed: {}", e)))?;
    
    Ok(result.to_string_lossy().into_owned())
}

/// Decode audio file to raw samples (16kHz mono Float32)
#[napi]
pub fn decode_audio(path: String) -> napi::Result<Vec<f64>> {
    let samples = audio::decode_audio_file(&path)
        .map_err(|e| napi::Error::from_reason(format!("Failed to decode audio: {}", e)))?;
    
    Ok(samples.iter().map(|&s| s as f64).collect())
}

/// Decode audio buffer to raw samples (16kHz mono Float32)
#[napi]
pub fn decode_audio_buffer(buffer: napi::bindgen_prelude::Buffer) -> napi::Result<Vec<f64>> {
    let samples = audio::decode_audio_buffer(&buffer)
        .map_err(|e| napi::Error::from_reason(format!("Failed to decode audio: {}", e)))?;
    
    Ok(samples.iter().map(|&s| s as f64).collect())
}

/// Format seconds to timestamp string (HH:MM:SS.mmm or MM:SS.mmm)
#[napi]
pub fn format_timestamp(seconds: f64, always_include_hours: Option<bool>) -> String {
    let include_hours = always_include_hours.unwrap_or(false);
    let hours = (seconds / 3600.0).floor() as u32;
    let minutes = ((seconds % 3600.0) / 60.0).floor() as u32;
    let secs = seconds % 60.0;
    
    if include_hours || hours > 0 {
        format!("{:02}:{:02}:{:06.3}", hours, minutes, secs)
    } else {
        format!("{:02}:{:06.3}", minutes, secs)
    }
}

/// Check if GPU acceleration is available (Metal on macOS, CUDA on Linux/Windows)
#[napi]
pub fn is_gpu_available() -> bool {
    is_metal_available() || is_cuda_available()
}

/// Get the number of available GPU devices
#[napi]
pub fn get_gpu_count() -> i32 {
    if is_metal_available() {
        1 // Metal is unified, typically 1 GPU
    } else if is_cuda_available() {
        1 // Would need CUDA API to get actual count
    } else {
        0
    }
}

/// Get the best available device ("metal", "cuda", or "cpu")
#[napi]
pub fn get_best_device() -> String {
    if is_metal_available() {
        "metal".to_string()
    } else if is_cuda_available() {
        "cuda".to_string()
    } else {
        "cpu".to_string()
    }
}

/// Check if Silero VAD (neural network voice activity detection) is available
/// Returns true if the silero-vad feature is enabled
#[napi]
pub fn is_silero_vad_available() -> bool {
    cfg!(feature = "silero-vad")
}

/// Get the path where the Silero VAD model would be stored
#[napi]
pub fn get_silero_model_path() -> String {
    download::default_cache_dir()
        .join("silero_vad.onnx")
        .to_string_lossy()
        .into_owned()
}

/// Download the Silero VAD model (async)
/// Returns the path to the downloaded model
#[napi]
pub async fn download_silero_model() -> napi::Result<String> {
    let cache_dir = download::default_cache_dir();
    let model_path = cache_dir.join("silero_vad.onnx");
    
    if model_path.exists() {
        return Ok(model_path.to_string_lossy().into_owned());
    }
    
    // Download from Silero VAD GitHub releases
    let url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx";
    
    // Ensure cache directory exists
    tokio::fs::create_dir_all(&cache_dir).await
        .map_err(|e| napi::Error::from_reason(format!("Failed to create cache dir: {}", e)))?;
    
    // Download the model
    let response = reqwest::get(url).await
        .map_err(|e| napi::Error::from_reason(format!("Failed to download Silero VAD model: {}", e)))?;
    
    if !response.status().is_success() {
        return Err(napi::Error::from_reason(format!(
            "Failed to download Silero VAD model: HTTP {}", response.status()
        )));
    }
    
    let bytes = response.bytes().await
        .map_err(|e| napi::Error::from_reason(format!("Failed to read response: {}", e)))?;
    
    tokio::fs::write(&model_path, &bytes).await
        .map_err(|e| napi::Error::from_reason(format!("Failed to write model file: {}", e)))?;
    
    Ok(model_path.to_string_lossy().into_owned())
}

// ============== Streaming Transcription ==============

use std::sync::Mutex;
use std::collections::HashMap;

/// A streaming transcription segment (stable or preview)
#[napi(object)]
#[derive(Clone, Debug)]
pub struct StreamingSegment {
    /// Segment text
    pub text: String,
    /// Start time in the audio stream (seconds)
    pub start: f64,
    /// End time in the audio stream (seconds) 
    pub end: f64,
    /// Whether this segment is final (won't change) or preview (may change)
    pub is_final: bool,
}

/// Result from processing streaming audio
#[napi(object)]
#[derive(Clone, Debug)]
pub struct StreamingResult {
    /// Stable (final) segments that won't change
    pub stable_segments: Vec<StreamingSegment>,
    /// Preview text that may change with more audio
    pub preview_text: Option<String>,
    /// Current buffer duration in seconds
    pub buffer_duration: f64,
    /// Total audio processed so far in seconds
    pub total_duration: f64,
}

/// Configuration for streaming transcription
#[napi(object)]
#[derive(Clone, Debug)]
pub struct StreamingOptions {
    /// Minimum buffer before transcription (seconds, default: 1.0)
    pub min_buffer_seconds: Option<f64>,
    /// Stability margin from buffer end (seconds, default: 1.5)
    pub stability_margin_seconds: Option<f64>,
    /// Context overlap to keep after committing (seconds, default: 0.5)
    pub context_overlap_seconds: Option<f64>,
    /// Maximum buffer size (seconds, default: 30.0)
    pub max_buffer_seconds: Option<f64>,
    /// Language for transcription
    pub language: Option<String>,
    /// Beam size (default: 5)
    pub beam_size: Option<u32>,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            min_buffer_seconds: None,
            stability_margin_seconds: None,
            context_overlap_seconds: None,
            max_buffer_seconds: None,
            language: None,
            beam_size: None,
        }
    }
}

/// Internal streaming session state
struct StreamingSessionState {
    /// Rolling audio buffer
    pub buffer: Vec<f32>,
    /// Total samples offset (discarded samples count)
    pub offset_samples: usize,
    /// Configuration
    pub min_buffer_samples: usize,
    pub stability_margin_samples: usize,
    pub context_overlap_samples: usize,
    pub max_buffer_samples: usize,
    pub language: Option<String>,
    pub beam_size: usize,
}

impl StreamingSessionState {
    fn new(opts: &StreamingOptions) -> Self {
        let sample_rate = 16000.0;
        Self {
            buffer: Vec::with_capacity(30 * 16000), // 30 seconds capacity
            offset_samples: 0,
            min_buffer_samples: (opts.min_buffer_seconds.unwrap_or(1.0) * sample_rate) as usize,
            stability_margin_samples: (opts.stability_margin_seconds.unwrap_or(1.5) * sample_rate) as usize,
            context_overlap_samples: (opts.context_overlap_seconds.unwrap_or(0.5) * sample_rate) as usize,
            max_buffer_samples: (opts.max_buffer_seconds.unwrap_or(30.0) * sample_rate) as usize,
            language: opts.language.clone(),
            beam_size: opts.beam_size.unwrap_or(5) as usize,
        }
    }

    fn add_samples(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
    }

    fn buffer_duration_seconds(&self) -> f64 {
        self.buffer.len() as f64 / 16000.0
    }

    fn total_duration_seconds(&self) -> f64 {
        (self.offset_samples + self.buffer.len()) as f64 / 16000.0
    }

    fn has_enough_audio(&self) -> bool {
        self.buffer.len() >= self.min_buffer_samples
    }

    fn is_buffer_full(&self) -> bool {
        self.buffer.len() >= self.max_buffer_samples
    }

    fn get_buffer(&self) -> &[f32] {
        &self.buffer
    }

    fn audio_offset_seconds(&self) -> f64 {
        self.offset_samples as f64 / 16000.0
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.offset_samples = 0;
    }
}

/// Streaming transcription engine with LocalAgreement algorithm
/// 
/// This enables true streaming transcription by:
/// 1. Maintaining a rolling audio buffer per session
/// 2. Running inference on overlapping windows
/// 3. Only emitting text that is "stable" (agreed upon across inference runs)
#[napi]
pub struct StreamingEngine {
    ctx: WhisperContext,
    num_threads: u32,
    sessions: Mutex<HashMap<i64, StreamingSessionState>>,
    next_session_id: Mutex<i64>,
}

#[napi]
impl StreamingEngine {
    /// Create a new streaming transcription engine
    #[napi(constructor)]
    pub fn new(model_path: String) -> napi::Result<Self> {
        Self::with_options(model_path, None)
    }

    /// Create a new streaming transcription engine with options
    #[napi(factory)]
    pub fn with_options(model_path: String, options: Option<ModelOptions>) -> napi::Result<Self> {
        let opts = options.unwrap_or_default();
        let resolved_path = download::resolve_model_path(&model_path);
        
        if !std::path::Path::new(&resolved_path).exists() {
            if download::get_repo_for_size(&model_path).is_some() {
                return Err(napi::Error::from_reason(format!(
                    "Model '{}' not found. Download it first using: await downloadModel('{}')",
                    model_path, model_path
                )));
            }
            return Err(napi::Error::from_reason(format!(
                "Model not found at: {}", resolved_path
            )));
        }
        
        let mut ctx_params = WhisperContextParameters::default();
        
        let use_gpu = match opts.device.as_deref() {
            Some("cpu") | Some("CPU") => false,
            Some("metal") | Some("Metal") | Some("METAL") => true,
            Some("cuda") | Some("CUDA") => true,
            Some("auto") | Some("AUTO") | None => {
                is_metal_available() || is_cuda_available()
            }
            _ => false,
        };
        ctx_params.use_gpu(use_gpu);
        
        let ctx = WhisperContext::new_with_params_and_state(&resolved_path, ctx_params)
            .map_err(|e| napi::Error::from_reason(format!("Failed to load model: {}", e)))?;
        
        let num_threads = opts.cpu_threads.unwrap_or(4);
        
        Ok(Self {
            ctx,
            num_threads,
            sessions: Mutex::new(HashMap::new()),
            next_session_id: Mutex::new(0),
        })
    }

    /// Create a new streaming session
    /// Returns the session ID
    #[napi]
    pub fn create_session(&self, options: Option<StreamingOptions>) -> napi::Result<i64> {
        let opts = options.unwrap_or_default();
        
        let mut next_id = self.next_session_id.lock()
            .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;
        let session_id = *next_id;
        *next_id += 1;
        
        let session = StreamingSessionState::new(&opts);
        
        let mut sessions = self.sessions.lock()
            .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;
        sessions.insert(session_id, session);
        
        Ok(session_id)
    }

    /// Add audio samples to a streaming session and process
    /// 
    /// Returns stable segments (final) and preview text (may change)
    #[napi]
    pub fn process_audio(&self, session_id: i64, samples: Vec<f64>) -> napi::Result<StreamingResult> {
        // Convert f64 to f32
        let samples_f32: Vec<f32> = samples.iter().map(|&x| x as f32).collect();
        
        let mut sessions = self.sessions.lock()
            .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;
        
        let session = sessions.get_mut(&session_id)
            .ok_or_else(|| napi::Error::from_reason(format!("Session {} not found", session_id)))?;
        
        // Add new samples to buffer
        session.add_samples(&samples_f32);
        
        // Check if we have enough audio to transcribe
        if !session.has_enough_audio() && !session.is_buffer_full() {
            return Ok(StreamingResult {
                stable_segments: vec![],
                preview_text: None,
                buffer_duration: session.buffer_duration_seconds(),
                total_duration: session.total_duration_seconds(),
            });
        }
        
        // Get buffer and transcribe
        let buffer = session.get_buffer().to_vec();
        let language = session.language.clone();
        let beam_size = session.beam_size;
        let buffer_duration = session.buffer_duration_seconds();
        let audio_offset = session.audio_offset_seconds();
        let stability_margin = session.stability_margin_samples as f64 / 16000.0;
        
        // Create state for transcription
        let mut state = self.ctx.create_state()
            .map_err(|e| napi::Error::from_reason(format!("Failed to create state: {}", e)))?;
        
        // Build params
        let strategy = if beam_size <= 1 {
            SamplingStrategy::Greedy { best_of: 1 }
        } else {
            SamplingStrategy::BeamSearch { 
                beam_size: beam_size as i32,
                patience: 1.0,
            }
        };
        
        let mut params = FullParams::new(strategy);
        params.set_n_threads(self.num_threads as i32);
        params.set_token_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        
        if let Some(ref lang) = language {
            params.set_language(Some(lang));
        }
        
        // Run transcription
        state.full(params, &buffer)
            .map_err(|e| napi::Error::from_reason(format!("Transcription failed: {}", e)))?;
        
        // Calculate stability cutoff (relative to buffer)
        let stability_cutoff = buffer_duration - stability_margin;
        
        let mut stable_segments = Vec::new();
        let mut preview_text = String::new();
        let mut last_stable_end_time = 0.0f64;
        
        // Process results using TOKEN-LEVEL stability (AlignAtt approximation)
        // Instead of checking entire segments, we check each token's t1 timestamp.
        // This allows emitting partial segments when only some tokens are stable.
        let num_segments = state.full_n_segments();
        
        for i in 0..num_segments {
            let segment = match state.get_segment(i) {
                Some(s) => s,
                None => continue,
            };
            
            let segment_start = segment.start_timestamp() as f64 / 100.0;
            let num_tokens = segment.n_tokens();
            
            if num_tokens == 0 {
                continue;
            }
            
            // Token-level stability check
            let mut stable_token_text = String::new();
            let mut preview_token_text = String::new();
            let mut last_stable_token_end = segment_start;
            let mut first_stable_token_start: Option<f64> = None;
            let mut hit_unstable = false;
            
            for j in 0..num_tokens {
                let token = match segment.get_token(j) {
                    Some(t) => t,
                    None => continue,
                };
                
                let token_text = match token.to_str_lossy() {
                    Ok(t) => t.into_owned(),
                    Err(_) => continue,
                };
                
                // Skip special tokens
                if token_text.starts_with('[') || token_text.starts_with('<') {
                    continue;
                }
                
                let token_data = token.token_data();
                let token_end = token_data.t1 as f64 / 100.0;  // centiseconds to seconds
                let token_start = token_data.t0 as f64 / 100.0;
                
                if !hit_unstable && stability_cutoff > 0.0 && token_end <= stability_cutoff {
                    // This token is stable
                    stable_token_text.push_str(&token_text);
                    last_stable_token_end = token_end;
                    if first_stable_token_start.is_none() {
                        first_stable_token_start = Some(token_start);
                    }
                } else {
                    // This token (and all following) are unstable
                    hit_unstable = true;
                    preview_token_text.push_str(&token_text);
                }
            }
            
            // Add stable tokens as a segment
            let stable_trimmed = stable_token_text.trim();
            if !stable_trimmed.is_empty() {
                let start_time = first_stable_token_start.unwrap_or(segment_start);
                stable_segments.push(StreamingSegment {
                    text: stable_trimmed.to_string(),
                    start: audio_offset + start_time,
                    end: audio_offset + last_stable_token_end,
                    is_final: true,
                });
                last_stable_end_time = last_stable_token_end;
            }
            
            // Add unstable tokens to preview
            let preview_trimmed = preview_token_text.trim();
            if !preview_trimmed.is_empty() {
                if !preview_text.is_empty() {
                    preview_text.push(' ');
                }
                preview_text.push_str(preview_trimmed);
            }
        }
        
        // Shift buffer: remove committed audio but keep overlap for context
        if last_stable_end_time > 0.0 {
            let context_overlap = session.context_overlap_samples as f64 / 16000.0;
            let drain_amount_time = if last_stable_end_time > context_overlap {
                last_stable_end_time - context_overlap
            } else {
                0.0
            };
            
            let drain_samples = (drain_amount_time * 16000.0) as usize;
            if drain_samples > 0 && drain_samples < session.buffer.len() {
                session.buffer.drain(0..drain_samples);
                session.offset_samples += drain_samples;
            }
        }
        
        // Handle max buffer overflow
        if session.is_buffer_full() && stable_segments.is_empty() {
            let force_drain = session.buffer.len() / 2;
            if force_drain > 0 {
                session.buffer.drain(0..force_drain);
                session.offset_samples += force_drain;
            }
        }
        
        Ok(StreamingResult {
            stable_segments,
            preview_text: if preview_text.is_empty() { None } else { Some(preview_text) },
            buffer_duration: session.buffer_duration_seconds(),
            total_duration: session.total_duration_seconds(),
        })
    }

    /// Flush session - return all remaining audio as final
    #[napi]
    pub fn flush_session(&self, session_id: i64) -> napi::Result<StreamingResult> {
        let mut sessions = self.sessions.lock()
            .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;
        
        let session = sessions.get_mut(&session_id)
            .ok_or_else(|| napi::Error::from_reason(format!("Session {} not found", session_id)))?;
        
        // Get remaining buffer
        let buffer = session.get_buffer().to_vec();
        
        if buffer.is_empty() {
            return Ok(StreamingResult {
                stable_segments: vec![],
                preview_text: None,
                buffer_duration: 0.0,
                total_duration: session.total_duration_seconds(),
            });
        }
        
        let language = session.language.clone();
        let beam_size = session.beam_size;
        let audio_offset = session.audio_offset_seconds();
        
        // Create state for transcription
        let mut state = self.ctx.create_state()
            .map_err(|e| napi::Error::from_reason(format!("Failed to create state: {}", e)))?;
        
        let strategy = if beam_size <= 1 {
            SamplingStrategy::Greedy { best_of: 1 }
        } else {
            SamplingStrategy::BeamSearch { 
                beam_size: beam_size as i32,
                patience: 1.0,
            }
        };
        
        let mut params = FullParams::new(strategy);
        params.set_n_threads(self.num_threads as i32);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        
        if let Some(ref lang) = language {
            params.set_language(Some(lang));
        }
        
        // Run transcription
        state.full(params, &buffer)
            .map_err(|e| napi::Error::from_reason(format!("Transcription failed: {}", e)))?;
        
        // All segments are final on flush
        let mut final_segments = Vec::new();
        let num_segments = state.full_n_segments();
        
        for i in 0..num_segments {
            let segment = match state.get_segment(i) {
                Some(s) => s,
                None => continue,
            };
            
            let text = match segment.to_str_lossy() {
                Ok(t) => t.into_owned(),
                Err(_) => continue,
            };
            
            let raw_text = text.trim();
            if raw_text.is_empty() {
                continue;
            }
            
            let start_ts = segment.start_timestamp();
            let end_ts = segment.end_timestamp();
            
            let segment_start = start_ts as f64 / 100.0;
            let segment_end = end_ts as f64 / 100.0;
            
            final_segments.push(StreamingSegment {
                text: raw_text.to_string(),
                start: audio_offset + segment_start,
                end: audio_offset + segment_end,
                is_final: true,
            });
        }
        
        let total_duration = session.total_duration_seconds();
        
        // Clear session
        session.reset();
        
        Ok(StreamingResult {
            stable_segments: final_segments,
            preview_text: None,
            buffer_duration: 0.0,
            total_duration,
        })
    }

    /// Reset a streaming session (clear buffer, keep session)
    #[napi]
    pub fn reset_session(&self, session_id: i64) -> napi::Result<()> {
        let mut sessions = self.sessions.lock()
            .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;
        
        let session = sessions.get_mut(&session_id)
            .ok_or_else(|| napi::Error::from_reason(format!("Session {} not found", session_id)))?;
        
        session.reset();
        Ok(())
    }

    /// Close a streaming session
    #[napi]
    pub fn close_session(&self, session_id: i64) -> napi::Result<()> {
        let mut sessions = self.sessions.lock()
            .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;
        
        sessions.remove(&session_id);
        Ok(())
    }

    /// Get the number of active sessions
    #[napi]
    pub fn session_count(&self) -> napi::Result<u32> {
        let sessions = self.sessions.lock()
            .map_err(|e| napi::Error::from_reason(format!("Lock error: {}", e)))?;
        Ok(sessions.len() as u32)
    }

    /// Get the expected sampling rate (16000 Hz for Whisper)
    #[napi]
    pub fn sampling_rate(&self) -> u32 {
        WHISPER_SAMPLE_RATE
    }
}
