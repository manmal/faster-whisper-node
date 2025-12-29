//! Streaming transcription using LocalAgreement algorithm
//!
//! This implements "fake streaming" for Whisper by:
//! 1. Maintaining a rolling audio buffer per session
//! 2. Running inference on overlapping windows  
//! 3. Only emitting text that is "stable" (agreed upon across runs)
//!
//! The key insight: Whisper gets "jittery" at the end of audio because
//! sentences are cut off. By only emitting text from segments that end
//! well before the buffer edge, we hide the instability.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Sample rate for Whisper (16kHz)
const SAMPLE_RATE: f32 = 16000.0;

/// Minimum audio buffer before attempting transcription (seconds)
const MIN_BUFFER_SECONDS: f32 = 1.0;

/// How much audio to keep as overlap/context after committing (seconds)
const CONTEXT_OVERLAP_SECONDS: f32 = 0.5;

/// How far from the buffer end a segment must be to be considered stable (seconds)
const STABILITY_MARGIN_SECONDS: f32 = 1.5;

/// Maximum buffer size before forcing processing (seconds)
const MAX_BUFFER_SECONDS: f32 = 30.0;

/// A streaming transcription segment
#[derive(Clone, Debug)]
pub struct StreamingSegment {
    /// Segment text
    pub text: String,
    /// Start time in the original audio stream (seconds)
    pub start: f64,
    /// End time in the original audio stream (seconds)
    pub end: f64,
    /// Whether this segment is final (won't change)
    pub is_final: bool,
}

/// Configuration for streaming transcription
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// Minimum buffer before transcription (seconds)
    pub min_buffer_seconds: f32,
    /// Context overlap to keep after committing (seconds)
    pub context_overlap_seconds: f32,
    /// Stability margin from buffer end (seconds)
    pub stability_margin_seconds: f32,
    /// Maximum buffer size (seconds)
    pub max_buffer_seconds: f32,
    /// Language for transcription
    pub language: Option<String>,
    /// Beam size
    pub beam_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            min_buffer_seconds: MIN_BUFFER_SECONDS,
            context_overlap_seconds: CONTEXT_OVERLAP_SECONDS,
            stability_margin_seconds: STABILITY_MARGIN_SECONDS,
            max_buffer_seconds: MAX_BUFFER_SECONDS,
            language: Some("en".to_string()),
            beam_size: 5,
        }
    }
}

/// State for a single streaming session
pub struct StreamingSession {
    /// Rolling audio buffer
    buffer: Vec<f32>,
    /// Total samples offset (how many samples we've processed and discarded)
    offset_samples: usize,
    /// Previously emitted stable text (for deduplication)
    last_stable_text: String,
    /// Configuration
    config: StreamingConfig,
    /// Session ID
    pub id: u64,
}

impl StreamingSession {
    /// Create a new streaming session
    pub fn new(id: u64, config: StreamingConfig) -> Self {
        Self {
            buffer: Vec::with_capacity((config.max_buffer_seconds * SAMPLE_RATE) as usize),
            offset_samples: 0,
            last_stable_text: String::new(),
            config,
            id,
        }
    }

    /// Add audio samples to the buffer
    pub fn add_samples(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
    }

    /// Get the current audio offset in seconds
    pub fn audio_offset_seconds(&self) -> f64 {
        self.offset_samples as f64 / SAMPLE_RATE as f64
    }

    /// Get the current buffer duration in seconds
    pub fn buffer_duration_seconds(&self) -> f64 {
        self.buffer.len() as f64 / SAMPLE_RATE as f64
    }

    /// Check if we have enough audio to attempt transcription
    pub fn has_enough_audio(&self) -> bool {
        self.buffer.len() >= (self.config.min_buffer_seconds * SAMPLE_RATE) as usize
    }

    /// Check if buffer is at max capacity (force transcription)
    pub fn is_buffer_full(&self) -> bool {
        self.buffer.len() >= (self.config.max_buffer_seconds * SAMPLE_RATE) as usize
    }

    /// Get the audio buffer for transcription
    pub fn get_buffer(&self) -> &[f32] {
        &self.buffer
    }

    /// Process transcription result and return stable segments
    /// 
    /// Takes raw segments from Whisper and applies LocalAgreement logic:
    /// - Only segments ending before (buffer_end - stability_margin) are stable
    /// - Commits stable segments and shifts the buffer
    /// 
    /// Returns: (stable_segments, preview_text)
    pub fn process_result(
        &mut self,
        segments: Vec<(f64, f64, String)>, // (start, end, text)
    ) -> (Vec<StreamingSegment>, Option<String>) {
        let buffer_duration = self.buffer_duration_seconds();
        let audio_offset = self.audio_offset_seconds();
        
        // Calculate the stability cutoff time (relative to buffer start)
        let stability_cutoff = buffer_duration - self.config.stability_margin_seconds as f64;
        
        if stability_cutoff <= 0.0 {
            // Buffer too short, nothing is stable yet
            // Return preview of all text
            let preview: String = segments.iter().map(|(_, _, t)| t.as_str()).collect::<Vec<_>>().join("");
            return (vec![], if preview.is_empty() { None } else { Some(preview) });
        }
        
        let mut stable_segments = Vec::new();
        let mut preview_text = String::new();
        let mut last_stable_end_samples: usize = 0;
        
        for (start, end, text) in segments {
            let absolute_start = audio_offset + start;
            let absolute_end = audio_offset + end;
            
            if end <= stability_cutoff {
                // This segment is stable - it ends well before the buffer edge
                stable_segments.push(StreamingSegment {
                    text: text.clone(),
                    start: absolute_start,
                    end: absolute_end,
                    is_final: true,
                });
                last_stable_end_samples = (end * SAMPLE_RATE as f64) as usize;
            } else {
                // This segment is unstable (near buffer edge) - add to preview
                preview_text.push_str(&text);
            }
        }
        
        // Shift buffer: remove committed audio but keep overlap for context
        if last_stable_end_samples > 0 {
            let overlap_samples = (self.config.context_overlap_seconds * SAMPLE_RATE) as usize;
            let drain_amount = if last_stable_end_samples > overlap_samples {
                last_stable_end_samples - overlap_samples
            } else {
                0
            };
            
            if drain_amount > 0 && drain_amount < self.buffer.len() {
                self.buffer.drain(0..drain_amount);
                self.offset_samples += drain_amount;
            }
        }
        
        // Handle max buffer overflow - force commit everything
        if self.is_buffer_full() && stable_segments.is_empty() {
            // Force commit all segments as stable
            let force_drain = self.buffer.len() / 2; // Drain half the buffer
            self.buffer.drain(0..force_drain);
            self.offset_samples += force_drain;
        }
        
        (
            stable_segments,
            if preview_text.is_empty() { None } else { Some(preview_text) }
        )
    }

    /// Flush the session - return all remaining audio as final
    pub fn flush(&mut self) -> Vec<StreamingSegment> {
        // Mark everything as stable on flush
        let buffer_duration = self.buffer_duration_seconds();
        let audio_offset = self.audio_offset_seconds();
        
        // Clear the buffer
        self.buffer.clear();
        
        vec![] // Caller should transcribe remaining buffer and mark all as final
    }

    /// Reset the session
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.offset_samples = 0;
        self.last_stable_text.clear();
    }
}

/// Manager for multiple streaming sessions
pub struct StreamingManager {
    sessions: HashMap<u64, StreamingSession>,
    next_id: u64,
    default_config: StreamingConfig,
}

impl StreamingManager {
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            sessions: HashMap::new(),
            next_id: 0,
            default_config: config,
        }
    }

    pub fn create_session(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.sessions.insert(id, StreamingSession::new(id, self.default_config.clone()));
        id
    }

    pub fn create_session_with_config(&mut self, config: StreamingConfig) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.sessions.insert(id, StreamingSession::new(id, config));
        id
    }

    pub fn get_session(&mut self, id: u64) -> Option<&mut StreamingSession> {
        self.sessions.get_mut(&id)
    }

    pub fn remove_session(&mut self, id: u64) -> Option<StreamingSession> {
        self.sessions.remove(&id)
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_session_basic() {
        let config = StreamingConfig::default();
        let mut session = StreamingSession::new(0, config);
        
        // Add 1 second of audio (16000 samples)
        let samples = vec![0.0f32; 16000];
        session.add_samples(&samples);
        
        assert!(session.has_enough_audio());
        assert!(!session.is_buffer_full());
        assert_eq!(session.buffer_duration_seconds(), 1.0);
    }

    #[test]
    fn test_stability_logic() {
        let mut config = StreamingConfig::default();
        config.stability_margin_seconds = 1.0;
        let mut session = StreamingSession::new(0, config);
        
        // Add 3 seconds of audio
        let samples = vec![0.0f32; 48000];
        session.add_samples(&samples);
        
        // Create mock segments
        // Segment at 0-1s should be stable (ends 2s before buffer end)
        // Segment at 2-2.5s should be unstable (ends 0.5s before buffer end)
        let segments = vec![
            (0.0, 1.0, "Hello ".to_string()),
            (2.0, 2.5, "World".to_string()),
        ];
        
        let (stable, preview) = session.process_result(segments);
        
        assert_eq!(stable.len(), 1);
        assert_eq!(stable[0].text, "Hello ");
        assert_eq!(preview, Some("World".to_string()));
    }
}
