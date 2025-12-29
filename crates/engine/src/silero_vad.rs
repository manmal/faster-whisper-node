//! Silero VAD - Neural network voice activity detection
//!
//! Uses the Silero VAD ONNX model for accurate speech detection.
//! Much more accurate than energy-based VAD, especially for:
//! - Quiet speech
//! - Noisy backgrounds
//! - Music vs speech distinction

#[cfg(feature = "silero-vad")]
use ort::session::{builder::GraphOptimizationLevel, Session};
#[cfg(feature = "silero-vad")]
use ort::value::Tensor;
use std::path::Path;

/// Sample rate expected by Silero VAD
pub const SILERO_SAMPLE_RATE: u32 = 16000;

/// Window size for Silero VAD (512 samples = 32ms at 16kHz)
pub const SILERO_WINDOW_SIZE: usize = 512;

/// Silero VAD configuration
#[derive(Clone, Debug)]
pub struct SileroVadConfig {
    /// Speech probability threshold (0.0 to 1.0, default: 0.5)
    pub threshold: f32,
    /// Minimum speech duration in milliseconds (default: 250)
    pub min_speech_duration_ms: u32,
    /// Minimum silence duration to split segments (default: 100)
    pub min_silence_duration_ms: u32,
    /// Padding around speech segments in milliseconds (default: 30)
    pub speech_pad_ms: u32,
}

impl Default for SileroVadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
        }
    }
}

/// Detected speech segment
#[derive(Clone, Debug)]
pub struct SpeechSegment {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
}

/// Silero VAD using ONNX Runtime 2.0
/// 
/// Model inputs:
/// - input: [batch, samples] - audio samples
/// - state: [2, batch, 128] - LSTM hidden state
/// - sr: [] - sample rate (scalar)
/// 
/// Model outputs:
/// - output: [batch, 1] - speech probability
/// - stateN: [2, batch, 128] - updated LSTM state
#[cfg(feature = "silero-vad")]
pub struct SileroVad {
    session: Session,
    /// LSTM state (combined h and c) - shape [2, 1, 128]
    state: Vec<f32>,
    config: SileroVadConfig,
}

#[cfg(feature = "silero-vad")]
impl SileroVad {
    /// Load Silero VAD model from ONNX file
    pub fn new(model_path: impl AsRef<Path>, config: SileroVadConfig) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        // Initialize state: [2, 1, 128] = 256 elements
        let state = vec![0.0f32; 2 * 1 * 128];

        Ok(Self { session, state, config })
    }

    /// Load Silero VAD from embedded model bytes
    pub fn from_bytes(model_bytes: &[u8], config: SileroVadConfig) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_memory(model_bytes)?;

        let state = vec![0.0f32; 2 * 1 * 128];

        Ok(Self { session, state, config })
    }

    /// Process audio chunk and return speech probability
    ///
    /// # Arguments
    /// * `samples` - Audio samples (16kHz, mono, f32 normalized to [-1, 1])
    ///
    /// # Returns
    /// Speech probability (0.0 to 1.0)
    pub fn process_chunk(&mut self, samples: &[f32]) -> ort::Result<f32> {
        // Silero expects exactly SILERO_WINDOW_SIZE samples
        let chunk: Vec<f32> = if samples.len() < SILERO_WINDOW_SIZE {
            let mut padded = vec![0.0f32; SILERO_WINDOW_SIZE];
            padded[..samples.len()].copy_from_slice(samples);
            padded
        } else {
            samples[..SILERO_WINDOW_SIZE].to_vec()
        };

        // Prepare inputs as Tensors
        // input: [1, 512]
        let input = Tensor::from_array(([1, SILERO_WINDOW_SIZE], chunk))?;
        // state: [2, 1, 128]
        let state_tensor = Tensor::from_array(([2, 1, 128], self.state.clone()))?;
        // sr: scalar (0-dimensional, but ort wants 1D with 1 element)
        let sr = Tensor::from_array(([1], vec![SILERO_SAMPLE_RATE as i64]))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input" => input,
            "state" => state_tensor,
            "sr" => sr,
        ])?;

        // Update state for next call
        if let Some(state_out) = outputs.get("stateN") {
            let (_shape, data) = state_out.try_extract_tensor::<f32>()?;
            if data.len() == self.state.len() {
                self.state.copy_from_slice(data);
            }
        }

        // Get speech probability
        let output = outputs.get("output")
            .ok_or_else(|| ort::Error::new("Missing output tensor"))?;
        let (_shape, data) = output.try_extract_tensor::<f32>()?;
        let prob = if !data.is_empty() { data[0] } else { 0.0 };

        Ok(prob)
    }

    /// Process audio and detect speech segments
    ///
    /// # Arguments
    /// * `samples` - Full audio (16kHz, mono, f32)
    ///
    /// # Returns
    /// Vector of detected speech segments
    pub fn detect_speech(&mut self, samples: &[f32]) -> ort::Result<Vec<SpeechSegment>> {
        self.reset();

        let window_samples = SILERO_WINDOW_SIZE;
        let min_speech_samples = (self.config.min_speech_duration_ms as usize * SILERO_SAMPLE_RATE as usize) / 1000;
        let min_silence_samples = (self.config.min_silence_duration_ms as usize * SILERO_SAMPLE_RATE as usize) / 1000;
        let pad_samples = (self.config.speech_pad_ms as usize * SILERO_SAMPLE_RATE as usize) / 1000;

        // Process in windows
        let mut speech_probs = Vec::new();
        for chunk in samples.chunks(window_samples) {
            let prob = self.process_chunk(chunk)?;
            speech_probs.push(prob);
        }

        // Find speech regions
        let is_speech: Vec<bool> = speech_probs.iter()
            .map(|&p| p > self.config.threshold)
            .collect();

        // Detect continuous speech segments
        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut speech_start = 0;
        let mut silence_count = 0;

        for (i, &speech) in is_speech.iter().enumerate() {
            if speech {
                if !in_speech {
                    speech_start = i;
                    in_speech = true;
                }
                silence_count = 0;
            } else if in_speech {
                silence_count += 1;
                let silence_duration = silence_count * window_samples;

                if silence_duration >= min_silence_samples {
                    // End of speech segment
                    let start_sample = speech_start * window_samples;
                    let end_sample = (i - silence_count + 1) * window_samples;

                    if end_sample - start_sample >= min_speech_samples {
                        segments.push((start_sample, end_sample));
                    }

                    in_speech = false;
                    silence_count = 0;
                }
            }
        }

        // Handle segment that extends to the end
        if in_speech {
            let start_sample = speech_start * window_samples;
            let end_sample = samples.len();

            if end_sample - start_sample >= min_speech_samples {
                segments.push((start_sample, end_sample));
            }
        }

        // Convert to SpeechSegment with padding
        let result: Vec<SpeechSegment> = segments
            .into_iter()
            .map(|(start, end)| {
                let padded_start = start.saturating_sub(pad_samples);
                let padded_end = (end + pad_samples).min(samples.len());

                SpeechSegment {
                    start: padded_start as f64 / SILERO_SAMPLE_RATE as f64,
                    end: padded_end as f64 / SILERO_SAMPLE_RATE as f64,
                }
            })
            .collect();

        Ok(result)
    }

    /// Filter audio to keep only speech segments
    ///
    /// # Returns
    /// (filtered_samples, offset_map) where offset_map maps filtered time to original time
    pub fn filter_audio(&mut self, samples: &[f32]) -> ort::Result<(Vec<f32>, Vec<(f64, f64)>)> {
        let segments = self.detect_speech(samples)?;

        if segments.is_empty() {
            return Ok((samples.to_vec(), vec![(0.0, 0.0)]));
        }

        let mut filtered = Vec::new();
        let mut offset_map = Vec::new();
        let mut current_offset = 0.0;

        for seg in &segments {
            let start_idx = (seg.start * SILERO_SAMPLE_RATE as f64) as usize;
            let end_idx = ((seg.end * SILERO_SAMPLE_RATE as f64) as usize).min(samples.len());

            offset_map.push((current_offset, seg.start));
            filtered.extend_from_slice(&samples[start_idx..end_idx]);
            current_offset += seg.end - seg.start;
        }

        Ok((filtered, offset_map))
    }

    /// Reset hidden state (call between unrelated audio streams)
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// Check if Silero VAD feature is enabled
pub fn is_silero_available() -> bool {
    cfg!(feature = "silero-vad")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SileroVadConfig::default();
        assert_eq!(config.threshold, 0.5);
        assert_eq!(config.min_speech_duration_ms, 250);
    }

    #[test]
    fn test_silero_available() {
        // Just verify the function works
        let _ = is_silero_available();
    }
}
