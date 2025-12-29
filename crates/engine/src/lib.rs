//! faster-whisper-node engine
//!
//! High-performance Whisper transcription for Node.js via CTranslate2.

mod audio;
mod download;
mod streaming;
mod vad;
mod word_timestamps;

use napi_derive::napi;
use ct2rs::{Whisper, WhisperOptions, Config};
use ct2rs::sys::{Device, ComputeType, get_device_count};

use vad::{EnergyVad, VadOptions as InternalVadOptions};
use word_timestamps::parse_timestamped_text;


/// Get the number of available CUDA devices
fn cuda_device_count() -> i32 {
    get_device_count(Device::CUDA)
}

/// Check if CUDA is available
fn is_cuda_available() -> bool {
    cuda_device_count() > 0
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
    /// VAD configuration options
    pub vad_options: Option<VadOptions>,
    /// Hallucination silence threshold in seconds (default: None)
    /// Segments with a silent duration longer than this will be considered hallucinations
    pub hallucination_silence_threshold: Option<f64>,
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
        }
    }
}

/// Model configuration options
#[napi(object)]
#[derive(Clone, Debug)]
pub struct ModelOptions {
    /// Device to use: "cpu" or "cuda" (default: "cpu")
    pub device: Option<String>,
    /// Compute type: "default", "auto", "int8", "int8_float16", "int16", "float16", "float32"
    pub compute_type: Option<String>,
    /// Number of CPU threads per replica (0 for auto)
    pub cpu_threads: Option<u32>,
    /// Custom cache directory for auto-downloaded models
    pub cache_dir: Option<String>,
}

impl Default for ModelOptions {
    fn default() -> Self {
        Self {
            device: None,
            compute_type: None,
            cpu_threads: None,
            cache_dir: None,
        }
    }
}

/// Transcription result containing all segments and metadata
#[napi(object)]
#[derive(Clone, Debug)]
pub struct TranscriptionResult {
    /// All transcribed segments
    pub segments: Vec<Segment>,
    /// Detected or specified language
    pub language: String,
    /// Language detection probability (0 if language was specified)
    pub language_probability: f64,
    /// Total audio duration in seconds
    pub duration: f64,
    /// Audio duration after VAD filtering (equals duration if VAD not used)
    pub duration_after_vad: f64,
    /// Full transcribed text (all segments joined)
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

/// Download progress information
#[napi(object)]
#[derive(Clone, Debug)]
pub struct DownloadProgress {
    /// Current progress percentage (0-100)
    pub percent: f64,
    /// Current file being downloaded
    pub current_file: String,
    /// Total files to download
    pub total_files: u32,
    /// Current file index
    pub current_index: u32,
}

#[napi]
pub struct Engine {
    model: Whisper,
    sampling_rate: u32,
}

#[napi]
impl Engine {
    /// Create a new transcription engine from a model path or size
    /// 
    /// # Arguments
    /// * `model_path` - Either a path to a CTranslate2 model directory, or a model size 
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
        
        let device = match opts.device.as_deref() {
            Some("cuda") | Some("CUDA") => Device::CUDA,
            Some("auto") | Some("AUTO") => {
                if is_cuda_available() {
                    Device::CUDA
                } else {
                    Device::CPU
                }
            }
            _ => Device::CPU,
        };
        
        let compute_type = match opts.compute_type.as_deref() {
            Some("auto") => ComputeType::AUTO,
            Some("int8") => ComputeType::INT8,
            Some("int8_float16") => ComputeType::INT8_FLOAT16,
            Some("int8_float32") => ComputeType::INT8_FLOAT32,
            Some("int16") => ComputeType::INT16,
            Some("float16") => ComputeType::FLOAT16,
            Some("float32") => ComputeType::FLOAT32,
            _ => ComputeType::DEFAULT,
        };
        
        let config = Config {
            device,
            compute_type,
            num_threads_per_replica: opts.cpu_threads.unwrap_or(0) as usize,
            ..Config::default()
        };
        
        let model = Whisper::new(&resolved_path, config)
            .map_err(|e| napi::Error::from_reason(format!("Failed to load model: {}", e)))?;
        
        let sampling_rate = model.sampling_rate() as u32;
        
        Ok(Self { model, sampling_rate })
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
        if !self.model.is_multilingual() {
            return Err(napi::Error::from_reason(
                "Language detection requires a multilingual model (not .en variants)"
            ));
        }
        
        let samples = audio::decode_audio_file(&audio_path)
            .map_err(|e| napi::Error::from_reason(format!("Failed to decode audio: {}", e)))?;
        
        // Use first 30 seconds max
        let max_samples = (30.0 * self.sampling_rate as f64) as usize;
        let detection_samples: &[f32] = if samples.len() > max_samples {
            &samples[..max_samples]
        } else {
            &samples
        };
        
        // Do a quick transcription with language=None to trigger auto-detection
        let opts = TranscribeOptions::default();
        let result = self.transcribe_samples_internal(detection_samples, &opts)?;
        
        // Language is detected automatically during transcription
        // For proper language detection we'd need direct access to the detection layer
        // For now, return the language from transcription
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
        if !self.model.is_multilingual() {
            return Err(napi::Error::from_reason(
                "Language detection requires a multilingual model (not .en variants)"
            ));
        }
        
        let samples = audio::decode_audio_buffer(&buffer)
            .map_err(|e| napi::Error::from_reason(format!("Failed to decode audio buffer: {}", e)))?;
        
        let max_samples = (30.0 * self.sampling_rate as f64) as usize;
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
        self.sampling_rate
    }

    /// Check if the model is multilingual
    #[napi]
    pub fn is_multilingual(&self) -> bool {
        self.model.is_multilingual()
    }

    /// Get the number of supported languages
    #[napi]
    pub fn num_languages(&self) -> u32 {
        self.model.num_languages() as u32
    }

    // Internal transcription implementation
    fn transcribe_samples_internal(
        &self,
        samples: &[f32],
        opts: &TranscribeOptions,
    ) -> napi::Result<TranscriptionResult> {
        // Calculate original duration
        let duration = samples.len() as f64 / self.sampling_rate as f64;
        
        // Apply VAD filtering if enabled
        let (processed_samples, vad_offset_map) = if opts.vad_filter.unwrap_or(false) {
            let vad_opts = self.build_vad_options(opts.vad_options.as_ref());
            let vad = EnergyVad::new(self.sampling_rate, vad_opts);
            vad.filter_audio(samples)
        } else {
            (samples.to_vec(), vec![(0.0, 0.0)])
        };
        
        let duration_after_vad = processed_samples.len() as f64 / self.sampling_rate as f64;
        
        // Build whisper options
        let whisper_opts = self.build_whisper_options(opts);
        
        // Determine if we want timestamps (needed for word-level or just segment-level)
        let want_word_timestamps = opts.word_timestamps.unwrap_or(false);
        let timestamp = want_word_timestamps;
        
        // Perform transcription
        let results = self.model.generate(
            &processed_samples,
            opts.language.as_deref(),
            timestamp,
            &whisper_opts,
        ).map_err(|e| napi::Error::from_reason(format!("Transcription failed: {}", e)))?;
        
        // Build segments from results
        let mut segments = Vec::new();
        let mut full_text = String::new();
        let samples_per_segment = self.model.n_samples();
        let use_vad = opts.vad_filter.unwrap_or(false);
        let hallucination_threshold = opts.hallucination_silence_threshold;
        
        for (idx, result) in results.iter().enumerate() {
            let raw_text = result.trim();
            if raw_text.is_empty() {
                continue;
            }
            
            // Calculate segment timing in filtered audio
            let filtered_segment_start = (idx * samples_per_segment) as f64 / self.sampling_rate as f64;
            let filtered_segment_end = ((idx + 1) * samples_per_segment) as f64 / self.sampling_rate as f64;
            let filtered_segment_end = filtered_segment_end.min(duration_after_vad);
            
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
            let (clean_text, words) = if want_word_timestamps {
                let timed_words = parse_timestamped_text(raw_text);
                let clean = word_timestamps::clean_transcript(raw_text);
                
                // Convert to Word structs and adjust for segment offset and VAD
                let words: Vec<Word> = timed_words.into_iter().map(|w| {
                    let word_start = if use_vad {
                        vad::restore_timestamp(filtered_segment_start + w.start, &vad_offset_map)
                    } else {
                        segment_start + w.start
                    };
                    let word_end = if use_vad {
                        vad::restore_timestamp(filtered_segment_start + w.end, &vad_offset_map)
                    } else {
                        segment_start + w.end
                    };
                    
                    Word {
                        word: w.word,
                        start: word_start,
                        end: word_end,
                        probability: w.probability,
                    }
                }).collect();
                
                (clean, Some(words))
            } else {
                (raw_text.to_string(), None)
            };
            
            // Hallucination detection: skip segments with too much silence
            if let Some(threshold) = hallucination_threshold {
                let segment_duration = segment_end - segment_start;
                let text_len = clean_text.split_whitespace().count();
                
                // Estimate speaking rate and check for hallucination
                // Normal speaking is ~150 words/minute = 2.5 words/sec
                // If segment duration per word is > threshold, likely hallucination
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
            full_text.push_str(&clean_text);
            
            segments.push(Segment {
                id: idx as u32,
                seek: (idx * samples_per_segment) as u32,
                start: segment_start,
                end: segment_end,
                text: clean_text,
                tokens: vec![], // ct2rs doesn't expose token IDs directly in generate()
                temperature: opts.temperature.unwrap_or(1.0),
                avg_logprob: 0.0, // Not available from ct2rs high-level API
                compression_ratio: 0.0,
                no_speech_prob: 0.0,
                words,
            });
        }
        
        Ok(TranscriptionResult {
            segments,
            language: opts.language.clone().unwrap_or_else(|| "auto".to_string()),
            language_probability: 0.0, // Would need low-level API for this
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

    // Helper: build WhisperOptions from TranscribeOptions
    fn build_whisper_options(&self, opts: &TranscribeOptions) -> WhisperOptions {
        let mut whisper_opts = WhisperOptions::default();
        
        if let Some(beam_size) = opts.beam_size {
            whisper_opts.beam_size = beam_size as usize;
        }
        if let Some(patience) = opts.patience {
            whisper_opts.patience = patience as f32;
        }
        if let Some(length_penalty) = opts.length_penalty {
            whisper_opts.length_penalty = length_penalty as f32;
        }
        if let Some(repetition_penalty) = opts.repetition_penalty {
            whisper_opts.repetition_penalty = repetition_penalty as f32;
        }
        if let Some(no_repeat_ngram_size) = opts.no_repeat_ngram_size {
            whisper_opts.no_repeat_ngram_size = no_repeat_ngram_size as usize;
        }
        if let Some(temperature) = opts.temperature {
            whisper_opts.sampling_temperature = temperature as f32;
        }
        if let Some(suppress_blank) = opts.suppress_blank {
            whisper_opts.suppress_blank = suppress_blank;
        }
        if let Some(max_length) = opts.max_length {
            whisper_opts.max_length = max_length as usize;
        }
        
        whisper_opts
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

/// Check if CUDA (GPU acceleration) is available
#[napi]
pub fn is_gpu_available() -> bool {
    is_cuda_available()
}

/// Get the number of available CUDA GPU devices
#[napi]
pub fn get_gpu_count() -> i32 {
    cuda_device_count()
}

/// Get the best available device ("cuda" if GPU available, otherwise "cpu")
#[napi]
pub fn get_best_device() -> String {
    if is_cuda_available() {
        "cuda".to_string()
    } else {
        "cpu".to_string()
    }
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
    /// Rolling audio buffer (pub for direct access in process_audio)
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

    /// Process transcription result with LocalAgreement algorithm
    fn process_result(&mut self, segments: Vec<Segment>) -> StreamingResult {
        let buffer_duration = self.buffer_duration_seconds();
        let audio_offset = self.audio_offset_seconds();
        let stability_margin_seconds = self.stability_margin_samples as f64 / 16000.0;
        
        // Calculate stability cutoff (relative to buffer start)
        let stability_cutoff = buffer_duration - stability_margin_seconds;
        
        let mut stable_segments = Vec::new();
        let mut preview_text = String::new();
        let mut last_stable_end_samples: usize = 0;
        
        for segment in segments {
            // Segment times are relative to buffer start
            let relative_start = segment.start;
            let relative_end = segment.end;
            
            // Absolute times in the full audio stream
            let absolute_start = audio_offset + relative_start;
            let absolute_end = audio_offset + relative_end;
            
            if stability_cutoff > 0.0 && relative_end <= stability_cutoff {
                // This segment is STABLE - ends well before buffer edge
                stable_segments.push(StreamingSegment {
                    text: segment.text.clone(),
                    start: absolute_start,
                    end: absolute_end,
                    is_final: true,
                });
                last_stable_end_samples = (relative_end * 16000.0) as usize;
            } else {
                // This segment is UNSTABLE (preview) - near buffer edge
                preview_text.push_str(&segment.text);
            }
        }
        
        // Shift buffer: remove committed audio but keep overlap for context
        if last_stable_end_samples > 0 {
            let drain_amount = if last_stable_end_samples > self.context_overlap_samples {
                last_stable_end_samples - self.context_overlap_samples
            } else {
                0
            };
            
            if drain_amount > 0 && drain_amount < self.buffer.len() {
                self.buffer.drain(0..drain_amount);
                self.offset_samples += drain_amount;
            }
        }
        
        // Handle max buffer overflow - force commit half the buffer
        if self.is_buffer_full() && stable_segments.is_empty() {
            let force_drain = self.buffer.len() / 2;
            if force_drain > 0 {
                self.buffer.drain(0..force_drain);
                self.offset_samples += force_drain;
            }
        }
        
        StreamingResult {
            stable_segments,
            preview_text: if preview_text.is_empty() { None } else { Some(preview_text) },
            buffer_duration: self.buffer_duration_seconds(),
            total_duration: self.total_duration_seconds(),
        }
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
    model: Whisper,
    sampling_rate: u32,
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
        
        let device = match opts.device.as_deref() {
            Some("cuda") | Some("CUDA") => Device::CUDA,
            Some("auto") | Some("AUTO") => {
                if is_cuda_available() { Device::CUDA } else { Device::CPU }
            }
            _ => Device::CPU,
        };
        
        let compute_type = match opts.compute_type.as_deref() {
            Some("auto") => ComputeType::AUTO,
            Some("int8") => ComputeType::INT8,
            Some("int8_float16") => ComputeType::INT8_FLOAT16,
            Some("int8_float32") => ComputeType::INT8_FLOAT32,
            Some("int16") => ComputeType::INT16,
            Some("float16") => ComputeType::FLOAT16,
            Some("float32") => ComputeType::FLOAT32,
            _ => ComputeType::DEFAULT,
        };
        
        let config = Config {
            device,
            compute_type,
            num_threads_per_replica: opts.cpu_threads.unwrap_or(0) as usize,
            ..Config::default()
        };
        
        let model = Whisper::new(&resolved_path, config)
            .map_err(|e| napi::Error::from_reason(format!("Failed to load model: {}", e)))?;
        
        let sampling_rate = model.sampling_rate() as u32;
        
        Ok(Self {
            model,
            sampling_rate,
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
        
        // Build whisper options
        let mut whisper_opts = WhisperOptions::default();
        whisper_opts.beam_size = beam_size;
        
        // Transcribe with timestamps
        let results = self.model.generate(
            &buffer,
            language.as_deref(),
            true, // Enable timestamps
            &whisper_opts,
        ).map_err(|e| napi::Error::from_reason(format!("Transcription failed: {}", e)))?;
        
        // Calculate stability cutoff (relative to buffer)
        let stability_cutoff = buffer_duration - stability_margin;
        
        let mut stable_segments = Vec::new();
        let mut preview_text = String::new();
        let mut last_stable_end_time = 0.0f64;
        
        // Process results using word timestamps for stability detection
        for (_idx, result) in results.iter().enumerate() {
            let raw_text = result.trim();
            if raw_text.is_empty() {
                continue;
            }
            
            // Parse word timestamps to get actual timing
            let words = word_timestamps::parse_timestamped_text(raw_text);
            
            if words.is_empty() {
                // No timestamps available, treat entire text as preview
                let clean = word_timestamps::clean_transcript(raw_text);
                preview_text.push_str(&clean);
                continue;
            }
            
            // Use word timestamps to split stable vs preview
            let mut stable_words = Vec::new();
            let mut preview_words = Vec::new();
            
            for word in words {
                // Word timing is relative to this segment's start (which is relative to buffer)
                if stability_cutoff > 0.0 && word.end <= stability_cutoff {
                    stable_words.push(word);
                } else {
                    preview_words.push(word);
                }
            }
            
            // Create stable segment from stable words
            if !stable_words.is_empty() {
                let first = stable_words.first().unwrap();
                let last = stable_words.last().unwrap();
                let text: String = stable_words.iter().map(|w| w.word.as_str()).collect::<Vec<_>>().join(" ");
                
                stable_segments.push(StreamingSegment {
                    text,
                    start: audio_offset + first.start,
                    end: audio_offset + last.end,
                    is_final: true,
                });
                
                last_stable_end_time = last.end;
            }
            
            // Collect preview words
            for word in preview_words {
                if !preview_text.is_empty() && !preview_text.ends_with(' ') {
                    preview_text.push(' ');
                }
                preview_text.push_str(&word.word);
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
        
        // Build whisper options
        let mut whisper_opts = WhisperOptions::default();
        whisper_opts.beam_size = beam_size;
        
        // Transcribe remaining buffer
        let results = self.model.generate(
            &buffer,
            language.as_deref(),
            true,
            &whisper_opts,
        ).map_err(|e| napi::Error::from_reason(format!("Transcription failed: {}", e)))?;
        
        // All segments are final on flush
        let mut final_segments = Vec::new();
        let samples_per_segment = self.model.n_samples();
        let buffer_duration = session.buffer_duration_seconds();
        
        for (idx, result) in results.iter().enumerate() {
            let raw_text = result.trim();
            if raw_text.is_empty() {
                continue;
            }
            
            let segment_start = (idx * samples_per_segment) as f64 / self.sampling_rate as f64;
            let segment_end = ((idx + 1) * samples_per_segment) as f64 / self.sampling_rate as f64;
            let segment_end = segment_end.min(buffer_duration);
            
            let clean_text = if raw_text.contains('<') && raw_text.contains('>') {
                word_timestamps::clean_transcript(raw_text)
            } else {
                raw_text.to_string()
            };
            
            final_segments.push(StreamingSegment {
                text: clean_text,
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
        self.sampling_rate
    }
}
