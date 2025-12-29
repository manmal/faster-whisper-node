//! Model downloading from HuggingFace Hub
//!
//! Downloads Whisper models in CTranslate2 format from the Systran organization.

use anyhow::{Context, Result};
use futures::StreamExt;
use reqwest::Client;
// sha2 for future checksum validation
// use sha2::{Sha256, Digest};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Model sizes and their HuggingFace repository names
const MODEL_REPOS: &[(&str, &str)] = &[
    ("tiny", "Systran/faster-whisper-tiny"),
    ("tiny.en", "Systran/faster-whisper-tiny.en"),
    ("base", "Systran/faster-whisper-base"),
    ("base.en", "Systran/faster-whisper-base.en"),
    ("small", "Systran/faster-whisper-small"),
    ("small.en", "Systran/faster-whisper-small.en"),
    ("medium", "Systran/faster-whisper-medium"),
    ("medium.en", "Systran/faster-whisper-medium.en"),
    ("large-v1", "Systran/faster-whisper-large-v1"),
    ("large-v2", "Systran/faster-whisper-large-v2"),
    ("large-v3", "Systran/faster-whisper-large-v3"),
    ("distil-large-v3", "Systran/faster-distil-whisper-large-v3"),
];

/// Required files for a CTranslate2 Whisper model
#[allow(dead_code)]
const REQUIRED_FILES: &[&str] = &[
    "model.bin",
    "config.json",
    "vocabulary.txt",      // some models use .txt
    "tokenizer.json",      // others use tokenizer.json
];

/// Get the default cache directory for models
pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("faster-whisper-node")
        .join("models")
}

/// Get the path for a specific model size
pub fn model_path_for_size(size: &str, cache_dir: Option<&Path>) -> PathBuf {
    let cache = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_cache_dir);
    cache.join(size)
}

/// Check if a model is already downloaded
pub fn is_model_downloaded(size: &str, cache_dir: Option<&Path>) -> bool {
    let model_dir = model_path_for_size(size, cache_dir);
    
    if !model_dir.exists() {
        return false;
    }
    
    // Check for model.bin (required) and config.json
    let has_model = model_dir.join("model.bin").exists();
    let has_config = model_dir.join("config.json").exists();
    let has_vocab = model_dir.join("vocabulary.txt").exists() 
        || model_dir.join("tokenizer.json").exists();
    
    has_model && has_config && has_vocab
}

/// Resolve a model path or size to an actual path
/// If a path is given, returns it directly
/// If a size alias is given, returns the cached model path (downloading if needed)
pub fn resolve_model_path(path_or_size: &str) -> String {
    // If it looks like a path, return as-is
    if path_or_size.contains('/') || path_or_size.contains('\\') 
        || Path::new(path_or_size).exists() {
        return path_or_size.to_string();
    }
    
    // It's a size alias, return the cache path
    model_path_for_size(path_or_size, None)
        .to_string_lossy()
        .into_owned()
}

/// Get the HuggingFace repository for a model size
pub fn get_repo_for_size(size: &str) -> Option<&'static str> {
    MODEL_REPOS.iter()
        .find(|(s, _)| *s == size)
        .map(|(_, repo)| *repo)
}

/// Default preprocessor config for Whisper models
/// This is required by ct2rs but not always present in HuggingFace repos
const DEFAULT_PREPROCESSOR_CONFIG: &str = r#"{
  "chunk_length": 30,
  "feature_extractor_type": "WhisperFeatureExtractor",
  "feature_size": 80,
  "hop_length": 160,
  "n_fft": 400,
  "n_samples": 480000,
  "nb_max_frames": 3000,
  "padding_side": "right",
  "padding_value": 0.0,
  "processor_class": "WhisperProcessor",
  "return_attention_mask": false,
  "sampling_rate": 16000
}"#;

/// Download a model from HuggingFace Hub
pub async fn download_model<F>(
    size: &str,
    cache_dir: Option<&Path>,
    progress_callback: Option<F>,
) -> Result<PathBuf>
where
    F: Fn(f64) + Send + Sync,
{
    let repo = get_repo_for_size(size)
        .context(format!("Unknown model size: {}. Available: {:?}", 
            size, 
            MODEL_REPOS.iter().map(|(s, _)| *s).collect::<Vec<_>>()))?;
    
    let model_dir = model_path_for_size(size, cache_dir);
    
    // Create directory
    fs::create_dir_all(&model_dir).await
        .context("Failed to create model directory")?;
    
    let client = Client::builder()
        .user_agent("faster-whisper-node/0.1")
        .build()
        .context("Failed to create HTTP client")?;
    
    // Get file list from the repo
    let files_to_download = get_files_to_download(&client, repo).await?;
    let total_files = files_to_download.len();
    
    for (idx, (filename, url, expected_size)) in files_to_download.iter().enumerate() {
        let file_path = model_dir.join(filename);
        
        // Skip if file exists with correct size
        if let Ok(metadata) = fs::metadata(&file_path).await {
            if *expected_size > 0 && metadata.len() == *expected_size {
                if let Some(ref cb) = progress_callback {
                    cb(((idx + 1) as f64 / total_files as f64) * 100.0);
                }
                continue;
            }
        }
        
        download_file(&client, url, &file_path, *expected_size, |file_progress| {
            if let Some(ref cb) = progress_callback {
                let overall = (idx as f64 + file_progress / 100.0) / total_files as f64;
                cb(overall * 100.0);
            }
        }).await?;
    }
    
    // Ensure preprocessor_config.json exists (required by ct2rs)
    let preprocessor_path = model_dir.join("preprocessor_config.json");
    if !preprocessor_path.exists() {
        fs::write(&preprocessor_path, DEFAULT_PREPROCESSOR_CONFIG).await
            .context("Failed to write preprocessor_config.json")?;
    }
    
    if let Some(ref cb) = progress_callback {
        cb(100.0);
    }
    
    Ok(model_dir)
}

/// Get list of files to download from a HuggingFace repository
async fn get_files_to_download(client: &Client, repo: &str) -> Result<Vec<(String, String, u64)>> {
    // HuggingFace Hub API to list files
    let api_url = format!("https://huggingface.co/api/models/{}/tree/main", repo);
    
    let response = client.get(&api_url)
        .send()
        .await
        .context("Failed to fetch file list")?;
    
    if !response.status().is_success() {
        anyhow::bail!("Failed to fetch file list: {}", response.status());
    }
    
    let files: Vec<serde_json::Value> = response.json().await
        .context("Failed to parse file list")?;
    
    let mut result = Vec::new();
    
    for file in files {
        if let (Some(path), Some(size)) = (
            file.get("path").and_then(|p| p.as_str()),
            file.get("size").and_then(|s| s.as_u64()),
        ) {
            // Only download essential model files
            if should_download_file(path) {
                let url = format!(
                    "https://huggingface.co/{}/resolve/main/{}",
                    repo, path
                );
                result.push((path.to_string(), url, size));
            }
        }
    }
    
    // Verify we have the essential files
    let has_model = result.iter().any(|(p, _, _)| p == "model.bin");
    let has_config = result.iter().any(|(p, _, _)| p == "config.json");
    
    if !has_model {
        anyhow::bail!("Repository missing model.bin");
    }
    if !has_config {
        anyhow::bail!("Repository missing config.json");
    }
    
    Ok(result)
}

/// Determine if a file should be downloaded
fn should_download_file(path: &str) -> bool {
    let essential = [
        "model.bin",
        "config.json",
        "vocabulary.txt",
        "vocabulary.json",
        "tokenizer.json",
        "preprocessor_config.json",  // Required by ct2rs for mel spectrogram
    ];
    
    essential.iter().any(|e| path == *e)
}

/// Download a single file with progress reporting
async fn download_file<F>(
    client: &Client,
    url: &str,
    dest: &Path,
    expected_size: u64,
    progress_callback: F,
) -> Result<()>
where
    F: Fn(f64),
{
    let response = client.get(url)
        .send()
        .await
        .context("Failed to start download")?;
    
    if !response.status().is_success() {
        anyhow::bail!("Download failed: {}", response.status());
    }
    
    let total_size = response.content_length().unwrap_or(expected_size);
    
    let mut file = fs::File::create(dest).await
        .context("Failed to create file")?;
    
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Failed to read chunk")?;
        file.write_all(&chunk).await
            .context("Failed to write chunk")?;
        
        downloaded += chunk.len() as u64;
        
        if total_size > 0 {
            progress_callback((downloaded as f64 / total_size as f64) * 100.0);
        }
    }
    
    file.flush().await?;
    
    Ok(())
}

/// List available model sizes
pub fn available_model_sizes() -> Vec<&'static str> {
    MODEL_REPOS.iter().map(|(s, _)| *s).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_cache_dir() {
        let dir = default_cache_dir();
        assert!(dir.to_string_lossy().contains("faster-whisper-node"));
    }
    
    #[test]
    fn test_resolve_model_path_existing() {
        let resolved = resolve_model_path("/some/path/model");
        assert_eq!(resolved, "/some/path/model");
    }
    
    #[test]
    fn test_resolve_model_path_size() {
        let resolved = resolve_model_path("tiny");
        assert!(resolved.contains("tiny"));
        assert!(resolved.contains("faster-whisper-node"));
    }
    
    #[test]
    fn test_get_repo() {
        assert_eq!(get_repo_for_size("tiny"), Some("Systran/faster-whisper-tiny"));
        assert_eq!(get_repo_for_size("large-v3"), Some("Systran/faster-whisper-large-v3"));
        assert_eq!(get_repo_for_size("invalid"), None);
    }
}
