//! faster-whisper-node engine
//!
//! High-performance Whisper transcription for Node.js via CTranslate2.

mod audio;
mod download;

use napi_derive::napi;
use ct2rs::{Whisper, WhisperOptions, Config};
use ct2rs::sys::{Device, ComputeType};


// Re-export download functions
pub use download::{
    default_cache_dir, 
    model_path_for_size, 
    is_model_downloaded,
    available_model_sizes,
};

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
        // Calculate duration
        let duration = samples.len() as f64 / self.sampling_rate as f64;
        
        // Build whisper options
        let whisper_opts = self.build_whisper_options(opts);
        
        // Determine if we want timestamps
        let timestamp = opts.word_timestamps.unwrap_or(false);
        
        // Perform transcription
        let results = self.model.generate(
            samples,
            opts.language.as_deref(),
            timestamp,
            &whisper_opts,
        ).map_err(|e| napi::Error::from_reason(format!("Transcription failed: {}", e)))?;
        
        // Build segments from results
        let mut segments = Vec::new();
        let mut full_text = String::new();
        let samples_per_segment = self.model.n_samples();
        
        for (idx, result) in results.iter().enumerate() {
            let text = result.trim().to_string();
            if !text.is_empty() {
                let segment_start = (idx * samples_per_segment) as f64 / self.sampling_rate as f64;
                let segment_end = ((idx + 1) * samples_per_segment) as f64 / self.sampling_rate as f64;
                let segment_end = segment_end.min(duration);
                
                if !full_text.is_empty() {
                    full_text.push(' ');
                }
                full_text.push_str(&text);
                
                segments.push(Segment {
                    id: idx as u32,
                    seek: (idx * samples_per_segment) as u32,
                    start: segment_start,
                    end: segment_end,
                    text,
                    tokens: vec![], // ct2rs doesn't expose token IDs directly in generate()
                    temperature: opts.temperature.unwrap_or(1.0),
                    avg_logprob: 0.0, // Not available from ct2rs high-level API
                    compression_ratio: 0.0,
                    no_speech_prob: 0.0,
                });
            }
        }
        
        Ok(TranscriptionResult {
            segments,
            language: opts.language.clone().unwrap_or_else(|| "auto".to_string()),
            language_probability: 0.0, // Would need low-level API for this
            duration,
            text: full_text,
        })
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
