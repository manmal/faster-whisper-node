//! Audio decoding and processing
//! 
//! Supports multiple audio formats via symphonia and resampling to 16kHz mono.

use anyhow::{Context, Result};
use rubato::{FftFixedIn, Resampler};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use std::fs::File;
use std::io::Cursor;

/// Target sample rate for Whisper
pub const TARGET_SAMPLE_RATE: u32 = 16000;

/// Decode audio from a file path to f32 samples at 16kHz mono
pub fn decode_audio_file(path: &str) -> Result<Vec<f32>> {
    let file = File::open(path).context("Failed to open audio file")?;
    let mut hint = Hint::new();
    
    // Extract extension for format hint
    if let Some(ext) = std::path::Path::new(path).extension() {
        hint.with_extension(&ext.to_string_lossy());
    }
    
    decode_audio_stream(Box::new(file), hint)
}

/// Decode audio from a buffer to f32 samples at 16kHz mono
pub fn decode_audio_buffer(buffer: &[u8]) -> Result<Vec<f32>> {
    let cursor = Cursor::new(buffer.to_vec());
    let hint = detect_format_from_magic(buffer)?;
    decode_audio_stream(Box::new(cursor), hint)
}

/// Detect audio format from magic bytes
fn detect_format_from_magic(buffer: &[u8]) -> Result<Hint> {
    let mut hint = Hint::new();
    
    if buffer.len() < 12 {
        anyhow::bail!("Audio buffer too small to detect format");
    }
    
    // Check magic bytes
    if &buffer[0..4] == b"RIFF" && &buffer[8..12] == b"WAVE" {
        hint.with_extension("wav");
    } else if &buffer[0..4] == b"fLaC" {
        hint.with_extension("flac");
    } else if &buffer[0..4] == b"OggS" {
        hint.with_extension("ogg");
    } else if &buffer[0..3] == b"ID3" || (buffer[0] == 0xFF && (buffer[1] & 0xE0) == 0xE0) {
        hint.with_extension("mp3");
    } else if buffer.len() >= 8 && &buffer[4..8] == b"ftyp" {
        hint.with_extension("m4a");
    } else {
        // Try to let symphonia figure it out
    }
    
    Ok(hint)
}

/// Core audio decoding from any media source
fn decode_audio_stream(
    source: Box<dyn symphonia::core::io::MediaSource>,
    hint: Hint,
) -> Result<Vec<f32>> {
    let mss = MediaSourceStream::new(source, Default::default());
    
    let format_opts = FormatOptions {
        enable_gapless: true,
        ..Default::default()
    };
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();
    
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .context("Failed to probe audio format")?;
    
    let mut format = probed.format;
    
    // Find the first audio track
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("No audio tracks found")?;
    
    let track_id = track.id;
    let codec_params = track.codec_params.clone();
    let source_sample_rate = codec_params.sample_rate.context("Missing sample rate")?;
    let channels = codec_params.channels.context("Missing channel info")?.count();
    
    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &decoder_opts)
        .context("Failed to create audio decoder")?;
    
    let mut samples: Vec<f32> = Vec::new();
    
    // Decode all packets
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e)) 
                if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e).context("Failed to read packet"),
        };
        
        if packet.track_id() != track_id {
            continue;
        }
        
        let decoded = decoder.decode(&packet).context("Failed to decode packet")?;
        
        // Convert to mono f32
        append_samples(&decoded, channels, &mut samples);
    }
    
    // Resample if needed
    if source_sample_rate != TARGET_SAMPLE_RATE {
        samples = resample(&samples, source_sample_rate, TARGET_SAMPLE_RATE)?;
    }
    
    Ok(samples)
}

/// Append decoded samples to output, converting to mono
fn append_samples(buffer: &AudioBufferRef, channels: usize, output: &mut Vec<f32>) {
    match buffer {
        AudioBufferRef::F32(buf) => {
            let frames = buf.frames();
            for frame in 0..frames {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame];
                }
                output.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S16(buf) => {
            let frames = buf.frames();
            for frame in 0..frames {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame] as f32 / 32768.0;
                }
                output.push(sum / channels as f32);
            }
        }
        AudioBufferRef::S32(buf) => {
            let frames = buf.frames();
            for frame in 0..frames {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += buf.chan(ch)[frame] as f32 / 2147483648.0;
                }
                output.push(sum / channels as f32);
            }
        }
        AudioBufferRef::U8(buf) => {
            let frames = buf.frames();
            for frame in 0..frames {
                let mut sum = 0.0f32;
                for ch in 0..channels {
                    sum += (buf.chan(ch)[frame] as f32 - 128.0) / 128.0;
                }
                output.push(sum / channels as f32);
            }
        }
        _ => {
            // For other formats, try to get as f32 if possible
            // This is a fallback - shouldn't happen often
        }
    }
}

/// Resample audio to target sample rate using rubato
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    
    let resample_ratio = to_rate as f64 / from_rate as f64;
    
    // Calculate chunk size based on source rate for efficiency
    let chunk_size = if from_rate >= 44100 { 1024 } else { 512 };
    
    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        2, // subchunks
        1, // channels (mono)
    ).context("Failed to create resampler")?;
    
    let output_frames = (samples.len() as f64 * resample_ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_frames);
    
    let input_frames_needed = resampler.input_frames_next();
    let mut pos = 0;
    
    // Process in chunks
    while pos < samples.len() {
        let chunk_end = (pos + input_frames_needed).min(samples.len());
        let mut chunk = samples[pos..chunk_end].to_vec();
        
        // Pad with zeros if needed for final chunk
        if chunk.len() < input_frames_needed {
            chunk.resize(input_frames_needed, 0.0);
        }
        
        let input = vec![chunk];
        let processed = resampler.process(&input, None)
            .context("Resampling failed")?;
        
        output.extend_from_slice(&processed[0]);
        pos += input_frames_needed;
    }
    
    // Trim to expected length
    output.truncate(output_frames);
    
    Ok(output)
}

/// Parse raw WAV buffer (legacy support, for simple WAV files)
#[allow(dead_code)]
pub fn parse_wav_buffer_simple(buffer: &[u8]) -> Result<Vec<f32>> {
    if buffer.len() < 44 {
        anyhow::bail!("File too small to be a valid WAV");
    }
    
    // Check for RIFF header
    if &buffer[0..4] != b"RIFF" || &buffer[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file");
    }
    
    // Parse fmt chunk
    let mut pos = 12;
    let mut num_channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    
    while pos + 8 <= buffer.len() {
        let chunk_id = &buffer[pos..pos+4];
        let chunk_size = u32::from_le_bytes([
            buffer[pos+4], buffer[pos+5], buffer[pos+6], buffer[pos+7]
        ]) as usize;
        
        if chunk_id == b"fmt " {
            if pos + 8 + 16 > buffer.len() {
                anyhow::bail!("Invalid fmt chunk");
            }
            let audio_format = u16::from_le_bytes([buffer[pos+8], buffer[pos+9]]);
            num_channels = u16::from_le_bytes([buffer[pos+10], buffer[pos+11]]);
            sample_rate = u32::from_le_bytes([
                buffer[pos+12], buffer[pos+13], buffer[pos+14], buffer[pos+15]
            ]);
            bits_per_sample = u16::from_le_bytes([buffer[pos+22], buffer[pos+23]]);
            
            if audio_format != 1 {
                anyhow::bail!("Only PCM format is supported (got format {})", audio_format);
            }
        } else if chunk_id == b"data" {
            let data_start = pos + 8;
            let data_end = (data_start + chunk_size).min(buffer.len());
            let data = &buffer[data_start..data_end];
            
            let samples = parse_pcm_samples(data, num_channels, bits_per_sample)?;
            
            // Resample if needed
            if sample_rate != TARGET_SAMPLE_RATE {
                return resample(&samples, sample_rate, TARGET_SAMPLE_RATE);
            }
            return Ok(samples);
        }
        
        pos += 8 + chunk_size;
        if chunk_size % 2 != 0 {
            pos += 1;
        }
    }
    
    anyhow::bail!("No data chunk found in WAV file")
}

/// Parse PCM samples from raw bytes
#[allow(dead_code)]
fn parse_pcm_samples(data: &[u8], channels: u16, bits_per_sample: u16) -> Result<Vec<f32>> {
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let frame_size = bytes_per_sample * channels as usize;
    
    let samples: Vec<f32> = match bits_per_sample {
        16 => {
            data.chunks(frame_size)
                .filter_map(|frame| {
                    if frame.len() >= bytes_per_sample {
                        if channels == 1 {
                            let sample = i16::from_le_bytes([frame[0], frame[1]]);
                            Some(sample as f32 / 32768.0)
                        } else if channels >= 2 && frame.len() >= 4 {
                            let left = i16::from_le_bytes([frame[0], frame[1]]) as f32;
                            let right = i16::from_le_bytes([frame[2], frame[3]]) as f32;
                            Some((left + right) / 2.0 / 32768.0)
                        } else {
                            let sample = i16::from_le_bytes([frame[0], frame[1]]);
                            Some(sample as f32 / 32768.0)
                        }
                    } else {
                        None
                    }
                })
                .collect()
        }
        24 => {
            data.chunks(frame_size)
                .filter_map(|frame| {
                    if frame.len() >= 3 {
                        let sample = ((frame[2] as i32) << 16 | (frame[1] as i32) << 8 | (frame[0] as i32)) as i32;
                        let sample = if sample & 0x800000 != 0 {
                            sample | !0xFFFFFF
                        } else {
                            sample
                        };
                        Some(sample as f32 / 8388608.0)
                    } else {
                        None
                    }
                })
                .collect()
        }
        32 => {
            data.chunks(frame_size)
                .filter_map(|frame| {
                    if frame.len() >= 4 {
                        let sample = f32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]);
                        Some(sample)
                    } else {
                        None
                    }
                })
                .collect()
        }
        8 => {
            data.chunks(frame_size)
                .filter_map(|frame| {
                    if !frame.is_empty() {
                        let sample = (frame[0] as i16 - 128) as f32 / 128.0;
                        Some(sample)
                    } else {
                        None
                    }
                })
                .collect()
        }
        _ => {
            anyhow::bail!("Unsupported bit depth: {} bits", bits_per_sample);
        }
    };
    
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detect_wav_format() {
        let wav_header = b"RIFFxxxxWAVE";
        let hint = detect_format_from_magic(wav_header).unwrap();
        // Can't directly check hint, but this should not fail
    }
    
    #[test]
    fn test_resample_passthrough() {
        let samples = vec![0.5, -0.5, 0.25, -0.25];
        let result = resample(&samples, 16000, 16000).unwrap();
        assert_eq!(result.len(), samples.len());
    }
}
