//! Voice Activity Detection (VAD)
//!
//! Provides speech/non-speech detection for audio preprocessing.
//! This helps reduce hallucinations and improves transcription quality
//! by filtering out silent portions of audio.



/// VAD configuration options
#[derive(Debug, Clone)]
pub struct VadOptions {
    /// Speech detection threshold (0.0 to 1.0, default: 0.5)
    /// Higher values require louder audio to be considered speech
    pub threshold: f32,
    /// Minimum speech duration in milliseconds (default: 250)
    pub min_speech_duration_ms: u32,
    /// Maximum speech duration in seconds (default: 30.0, matches Whisper chunk size)
    pub max_speech_duration_s: f32,
    /// Minimum silence duration in milliseconds to split segments (default: 2000)
    pub min_silence_duration_ms: u32,
    /// Analysis window size in milliseconds (default: 30)
    pub window_size_ms: u32,
    /// Padding around speech segments in milliseconds (default: 400)
    pub speech_pad_ms: u32,
}

impl Default for VadOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            max_speech_duration_s: 30.0,
            min_silence_duration_ms: 2000,
            window_size_ms: 30,
            speech_pad_ms: 400,
        }
    }
}

/// A detected speech segment
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
}

impl SpeechSegment {
    /// Get the duration of this segment in seconds
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Simple energy-based Voice Activity Detector
///
/// This is a basic implementation that uses RMS energy to detect speech.
/// For production use, consider using Silero VAD via ONNX.
pub struct EnergyVad {
    options: VadOptions,
    sample_rate: u32,
}

impl EnergyVad {
    /// Create a new energy-based VAD
    pub fn new(sample_rate: u32, options: VadOptions) -> Self {
        Self { options, sample_rate }
    }
    
    /// Detect speech segments in audio samples
    ///
    /// # Arguments
    /// * `samples` - Audio samples normalized to [-1, 1]
    ///
    /// # Returns
    /// Vector of detected speech segments
    pub fn detect(&self, samples: &[f32]) -> Vec<SpeechSegment> {
        let window_samples = (self.options.window_size_ms as usize * self.sample_rate as usize) / 1000;
        let min_speech_samples = (self.options.min_speech_duration_ms as usize * self.sample_rate as usize) / 1000;
        let min_silence_samples = (self.options.min_silence_duration_ms as usize * self.sample_rate as usize) / 1000;
        let pad_samples = (self.options.speech_pad_ms as usize * self.sample_rate as usize) / 1000;
        let _max_speech_samples = (self.options.max_speech_duration_s * self.sample_rate as f32) as usize;
        
        // Calculate RMS energy for each window
        let energies: Vec<f32> = samples
            .chunks(window_samples)
            .map(|chunk| {
                let sum_sq: f32 = chunk.iter().map(|&s| s * s).sum();
                (sum_sq / chunk.len() as f32).sqrt()
            })
            .collect();
        
        if energies.is_empty() {
            return vec![];
        }
        
        // Normalize energies to [0, 1]
        let max_energy = energies.iter().cloned().fold(0.0f32, f32::max);
        if max_energy < 1e-10 {
            return vec![]; // Silent audio
        }
        
        let normalized: Vec<f32> = energies.iter().map(|e| e / max_energy).collect();
        
        // Detect speech/non-speech regions
        let is_speech: Vec<bool> = normalized.iter()
            .map(|&e| e > self.options.threshold)
            .collect();
        
        // Find continuous speech regions
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
        
        // Add padding and convert to SpeechSegment
        let mut result: Vec<SpeechSegment> = segments
            .into_iter()
            .map(|(start, end)| {
                let padded_start = start.saturating_sub(pad_samples);
                let padded_end = (end + pad_samples).min(samples.len());
                
                SpeechSegment {
                    start: padded_start as f64 / self.sample_rate as f64,
                    end: padded_end as f64 / self.sample_rate as f64,
                }
            })
            .collect();
        
        // Merge overlapping segments
        result = merge_overlapping_segments(result);
        
        // Split segments that exceed max duration
        result = split_long_segments(result, self.options.max_speech_duration_s as f64);
        
        result
    }
    
    /// Get audio samples for detected speech segments only
    ///
    /// Returns (filtered_samples, time_offset_map) where time_offset_map
    /// can be used to restore original timestamps
    pub fn filter_audio(&self, samples: &[f32]) -> (Vec<f32>, Vec<(f64, f64)>) {
        let segments = self.detect(samples);
        
        if segments.is_empty() {
            return (samples.to_vec(), vec![(0.0, 0.0)]);
        }
        
        let mut filtered = Vec::new();
        let mut offset_map = Vec::new();
        let mut current_offset = 0.0;
        
        for seg in &segments {
            let start_idx = (seg.start * self.sample_rate as f64) as usize;
            let end_idx = ((seg.end * self.sample_rate as f64) as usize).min(samples.len());
            
            // Map: (filtered_time, original_time)
            offset_map.push((current_offset, seg.start));
            
            filtered.extend_from_slice(&samples[start_idx..end_idx]);
            current_offset += seg.end - seg.start;
        }
        
        (filtered, offset_map)
    }
}

/// Merge overlapping speech segments
fn merge_overlapping_segments(mut segments: Vec<SpeechSegment>) -> Vec<SpeechSegment> {
    if segments.len() <= 1 {
        return segments;
    }
    
    // Sort by start time
    segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    
    let mut merged = Vec::new();
    let mut current = segments[0].clone();
    
    for seg in segments.into_iter().skip(1) {
        if seg.start <= current.end {
            // Overlapping, merge
            current.end = current.end.max(seg.end);
        } else {
            // Non-overlapping, push current and start new
            merged.push(current);
            current = seg;
        }
    }
    merged.push(current);
    
    merged
}

/// Split segments longer than max duration
fn split_long_segments(segments: Vec<SpeechSegment>, max_duration: f64) -> Vec<SpeechSegment> {
    let mut result = Vec::new();
    
    for seg in segments {
        if seg.duration() <= max_duration {
            result.push(seg);
        } else {
            // Split into chunks
            let mut current_start = seg.start;
            while current_start < seg.end {
                let chunk_end = (current_start + max_duration).min(seg.end);
                result.push(SpeechSegment {
                    start: current_start,
                    end: chunk_end,
                });
                current_start = chunk_end;
            }
        }
    }
    
    result
}

/// Restore original timestamps from filtered audio timestamps
///
/// Given a time in the filtered audio and the offset map,
/// returns the corresponding time in the original audio
pub fn restore_timestamp(filtered_time: f64, offset_map: &[(f64, f64)]) -> f64 {
    // Find the segment containing this time
    for i in (0..offset_map.len()).rev() {
        let (filtered_start, original_start) = offset_map[i];
        if filtered_time >= filtered_start {
            let offset_in_segment = filtered_time - filtered_start;
            return original_start + offset_in_segment;
        }
    }
    
    filtered_time
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_energy_vad_silent() {
        let vad = EnergyVad::new(16000, VadOptions::default());
        let silent = vec![0.0f32; 16000]; // 1 second of silence
        
        let segments = vad.detect(&silent);
        assert!(segments.is_empty());
    }
    
    #[test]
    fn test_energy_vad_speech() {
        let vad = EnergyVad::new(16000, VadOptions {
            threshold: 0.3,
            min_speech_duration_ms: 100,
            min_silence_duration_ms: 200,
            ..Default::default()
        });
        
        // Create audio with speech in the middle
        let mut samples = vec![0.0f32; 48000]; // 3 seconds
        
        // Add "speech" (sine wave) from 0.5s to 2.0s
        let speech_start = 8000; // 0.5s
        let speech_end = 32000; // 2.0s
        for i in speech_start..speech_end {
            samples[i] = (i as f32 * 0.1).sin() * 0.5;
        }
        
        let segments = vad.detect(&samples);
        
        assert!(!segments.is_empty());
        assert!(segments[0].start < 1.0);
        assert!(segments[0].end > 1.5);
    }
    
    #[test]
    fn test_merge_overlapping() {
        let segments = vec![
            SpeechSegment { start: 0.0, end: 1.0 },
            SpeechSegment { start: 0.5, end: 1.5 },
            SpeechSegment { start: 2.0, end: 3.0 },
        ];
        
        let merged = merge_overlapping_segments(segments);
        
        assert_eq!(merged.len(), 2);
        assert!((merged[0].start - 0.0).abs() < 0.01);
        assert!((merged[0].end - 1.5).abs() < 0.01);
        assert!((merged[1].start - 2.0).abs() < 0.01);
        assert!((merged[1].end - 3.0).abs() < 0.01);
    }
    
    #[test]
    fn test_restore_timestamp() {
        // Original audio: 0-1s (speech), 1-2s (silence), 2-3s (speech)
        // Filtered audio: 0-1s (first speech), 1-2s (second speech, was 2-3s)
        let offset_map = vec![
            (0.0, 0.0),   // First segment: filtered 0s = original 0s
            (1.0, 2.0),   // Second segment: filtered 1s = original 2s
        ];
        
        assert!((restore_timestamp(0.5, &offset_map) - 0.5).abs() < 0.01);
        assert!((restore_timestamp(1.5, &offset_map) - 2.5).abs() < 0.01);
    }
}
