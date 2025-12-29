//! Word-level timestamp extraction
//!
//! Parses timestamp tokens from Whisper output to extract word-level timing information.


use regex::Regex;
use std::sync::LazyLock;

/// Word with timing information
#[derive(Debug, Clone)]
pub struct TimedWord {
    /// The word text
    pub word: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds  
    pub end: f64,
    /// Confidence/probability (if available)
    pub probability: f64,
}

/// Timestamp token pattern: <|0.00|>, <|12.34|>, etc.
static TIMESTAMP_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<\|(\d+\.\d{2})\|>").expect("Invalid timestamp regex")
});

/// Special tokens to filter out
static SPECIAL_TOKEN_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<\|[^>]+\|>").expect("Invalid special token regex")
});

/// Parse a timestamped transcript into words with timing
///
/// Input format: "<|0.00|>Hello<|0.50|> world<|1.00|>"
/// Returns words with their start/end times
pub fn parse_timestamped_text(text: &str) -> Vec<TimedWord> {
    let mut words = Vec::new();
    let mut current_time: Option<f64> = None;
    let mut last_pos = 0;
    
    // Find all timestamp positions
    let timestamps: Vec<(usize, usize, f64)> = TIMESTAMP_PATTERN
        .captures_iter(text)
        .filter_map(|cap| {
            let m = cap.get(0)?;
            let time_str = cap.get(1)?.as_str();
            let time: f64 = time_str.parse().ok()?;
            Some((m.start(), m.end(), time))
        })
        .collect();
    
    if timestamps.is_empty() {
        // No timestamps, return the whole text as one word
        let clean_text = SPECIAL_TOKEN_PATTERN.replace_all(text, "").trim().to_string();
        if !clean_text.is_empty() {
            return vec![TimedWord {
                word: clean_text,
                start: 0.0,
                end: 0.0,
                probability: 1.0,
            }];
        }
        return words;
    }
    
    // Process segments between timestamps
    for (_i, (ts_start, ts_end, time)) in timestamps.iter().enumerate() {
        // Get text before this timestamp (since last position)
        if *ts_start > last_pos {
            let segment = &text[last_pos..*ts_start];
            let segment = SPECIAL_TOKEN_PATTERN.replace_all(segment, "").trim().to_string();
            
            if !segment.is_empty() && current_time.is_some() {
                // Split into individual words
                for word in segment.split_whitespace() {
                    let word = word.trim();
                    if !word.is_empty() {
                        words.push(TimedWord {
                            word: word.to_string(),
                            start: current_time.unwrap(),
                            end: *time,
                            probability: 1.0,
                        });
                    }
                }
            }
        }
        
        current_time = Some(*time);
        last_pos = *ts_end;
    }
    
    // Handle any remaining text after the last timestamp
    if last_pos < text.len() {
        let segment = &text[last_pos..];
        let segment = SPECIAL_TOKEN_PATTERN.replace_all(segment, "").trim().to_string();
        
        if !segment.is_empty() && current_time.is_some() {
            let end_time = current_time.unwrap() + 0.5; // Estimate end time
            for word in segment.split_whitespace() {
                let word = word.trim();
                if !word.is_empty() {
                    words.push(TimedWord {
                        word: word.to_string(),
                        start: current_time.unwrap(),
                        end: end_time,
                        probability: 1.0,
                    });
                }
            }
        }
    }
    
    // Distribute time evenly among words in each segment
    interpolate_word_times(&mut words);
    
    words
}

/// Interpolate word times more accurately when multiple words share the same timestamps
fn interpolate_word_times(words: &mut [TimedWord]) {
    if words.len() <= 1 {
        return;
    }
    
    let mut i = 0;
    while i < words.len() {
        // Find consecutive words with the same start/end times
        let start = words[i].start;
        let end = words[i].end;
        let mut j = i + 1;
        
        while j < words.len() && words[j].start == start && words[j].end == end {
            j += 1;
        }
        
        // If multiple words share the same segment, distribute time evenly
        if j > i + 1 {
            let count = (j - i) as f64;
            let duration = end - start;
            let word_duration = duration / count;
            
            for (idx, word) in words[i..j].iter_mut().enumerate() {
                word.start = start + (idx as f64) * word_duration;
                word.end = start + ((idx + 1) as f64) * word_duration;
            }
        }
        
        i = j;
    }
}

/// Clean a transcript by removing timestamp and special tokens
pub fn clean_transcript(text: &str) -> String {
    SPECIAL_TOKEN_PATTERN.replace_all(text, " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_timestamped_text() {
        let text = "<|0.00|>Hello<|0.50|> world<|1.00|>";
        let words = parse_timestamped_text(text);
        
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].word, "Hello");
        assert!((words[0].start - 0.0).abs() < 0.01);
        assert!((words[0].end - 0.50).abs() < 0.01);
        
        assert_eq!(words[1].word, "world");
        assert!((words[1].start - 0.50).abs() < 0.01);
        assert!((words[1].end - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_clean_transcript() {
        let text = "<|startoftranscript|><|en|><|transcribe|><|0.00|>Hello world<|1.00|><|endoftext|>";
        let clean = clean_transcript(text);
        assert_eq!(clean, "Hello world");
    }
    
    #[test]
    fn test_no_timestamps() {
        let text = "Hello world without timestamps";
        let words = parse_timestamped_text(text);
        
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "Hello world without timestamps");
    }
    
    #[test]
    fn test_interpolate_times() {
        let text = "<|0.00|>The quick brown fox<|2.00|>";
        let words = parse_timestamped_text(text);
        
        assert_eq!(words.len(), 4);
        
        // Times should be evenly distributed
        assert!((words[0].start - 0.0).abs() < 0.01);
        assert!((words[0].end - 0.5).abs() < 0.01);
        
        assert!((words[1].start - 0.5).abs() < 0.01);
        assert!((words[1].end - 1.0).abs() < 0.01);
        
        assert!((words[2].start - 1.0).abs() < 0.01);
        assert!((words[2].end - 1.5).abs() < 0.01);
        
        assert!((words[3].start - 1.5).abs() < 0.01);
        assert!((words[3].end - 2.0).abs() < 0.01);
    }
}
