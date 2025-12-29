const path = require('path');
const os = require('os');

// Detect platform and architecture
const platform = os.platform();
const arch = os.arch();

let binaryName;
if (platform === 'darwin' && arch === 'arm64') {
  binaryName = 'whisper-live-engine.darwin-arm64.node';
} else if (platform === 'darwin' && arch === 'x64') {
  binaryName = 'whisper-live-engine.darwin-x64.node';
} else if (platform === 'linux' && arch === 'x64') {
  binaryName = 'whisper-live-engine.linux-x64-gnu.node';
} else if (platform === 'linux' && arch === 'arm64') {
  binaryName = 'whisper-live-engine.linux-arm64-gnu.node';
} else if (platform === 'win32' && arch === 'x64') {
  binaryName = 'whisper-live-engine.win32-x64-msvc.node';
} else {
  throw new Error(`Unsupported platform: ${platform}-${arch}`);
}

// Load the native module from crates/engine
const nativeModule = require(`./crates/engine/${binaryName}`);

// Export Engine and StreamingEngine classes
const { Engine, StreamingEngine } = nativeModule;

// Export utility functions
const { 
  availableModels, 
  formatTimestamp,
  // Phase 2 additions
  isModelAvailable,
  getModelPath,
  getCacheDir,
  downloadModel,
  decodeAudio,
  decodeAudioBuffer,
  // Phase 4 additions
  isGpuAvailable,
  getGpuCount,
  getBestDevice,
} = nativeModule;

module.exports = { 
  Engine,
  // Streaming transcription (LocalAgreement algorithm)
  StreamingEngine,
  // Utility functions
  availableModels,
  formatTimestamp,
  // Model management
  isModelAvailable,
  getModelPath,
  getCacheDir,
  downloadModel,
  // Audio decoding
  decodeAudio,
  decodeAudioBuffer,
  // GPU detection
  isGpuAvailable,
  getGpuCount,
  getBestDevice,
};

// Add streaming utilities after export (avoid circular dependency)
// These are standalone modules that import from this file
module.exports.StreamingBatcher = require('./streaming-batcher.js').StreamingBatcher;
module.exports.createBatcher = require('./streaming-batcher.js').createBatcher;
module.exports.WorkerPoolBatcher = require('./worker-pool-batcher.js').WorkerPoolBatcher;
module.exports.createWorkerPool = require('./worker-pool-batcher.js').createWorkerPool;
