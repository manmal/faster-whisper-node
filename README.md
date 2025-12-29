# faster-whisper-node

A pure Node.js/Rust module for Whisper speech-to-text transcription. **No Python runtime required.**

Uses [CTranslate2](https://github.com/OpenNMT/CTranslate2) as the inference engine, the same battle-tested backend that powers [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

[![CI](https://github.com/manmal/faster-whisper-node/actions/workflows/ci.yml/badge.svg)](https://github.com/manmal/faster-whisper-node/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/faster-whisper-node.svg)](https://www.npmjs.com/package/faster-whisper-node)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js Version](https://img.shields.io/node/v/faster-whisper-node.svg)](https://nodejs.org)

## Features

- **Auto-download models** — Just specify a model name, it downloads automatically
- **Multi-format audio** — WAV, MP3, FLAC, OGG, and more (via symphonia)
- **Structured output** — Get segments with timestamps, confidence scores, and metadata
- **Word-level timestamps** — Get precise timing for each word
- **Voice Activity Detection** — Filter out silent portions of audio
- **Hallucination detection** — Skip segments that appear to be hallucinations
- **GPU auto-detection** — Automatically use CUDA if available
- **Transcription options** — Beam size, temperature, language, and more
- **Multiple input formats** — File paths, Buffers, or raw samples
- **Stereo support** — Automatically converts stereo to mono and resamples to 16kHz
- **No Python** — Pure Node.js/Rust, zero Python dependencies

## Quick Start

```javascript
const { Engine, downloadModel, formatTimestamp } = require('faster-whisper-node');

// Download a model (one-time, cached in ~/.cache/faster-whisper-node)
await downloadModel('tiny');

// Load model by name
const engine = new Engine('tiny');

// Simple transcription (returns text)
const text = engine.transcribe('./audio.wav');
console.log(text); // "Hello world."

// Transcribe MP3, FLAC, OGG, etc.
const text2 = engine.transcribe('./audio.mp3');

// Get segments with timestamps
const result = engine.transcribeFile('./audio.wav');
for (const segment of result.segments) {
  console.log(`[${formatTimestamp(segment.start)} -> ${formatTimestamp(segment.end)}] ${segment.text}`);
}
// [00:00.000 -> 00:02.500] Hello world.

// With options
const result2 = engine.transcribeFile('./audio.wav', {
  language: 'en',
  beamSize: 5,
  temperature: 0.0
});

// Word-level timestamps
const result3 = engine.transcribeFile('./audio.wav', {
  wordTimestamps: true
});
for (const segment of result3.segments) {
  if (segment.words) {
    for (const word of segment.words) {
      console.log(`${word.word} @ ${formatTimestamp(word.start)}`);
    }
  }
}

// VAD filtering (skip silent portions)
const result4 = engine.transcribeFile('./audio.wav', {
  vadFilter: true,
  vadOptions: {
    threshold: 0.5,
    minSpeechDurationMs: 250
  }
});

// GPU auto-detection
const { isGpuAvailable, getBestDevice } = require('faster-whisper-node');
console.log(`GPU available: ${isGpuAvailable()}`);
console.log(`Best device: ${getBestDevice()}`);

const gpuEngine = Engine.withOptions('tiny', { device: 'auto' }); // Uses GPU if available
```

---

## Why This Exists

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) is an excellent Python library that provides fast Whisper inference via CTranslate2. However, integrating Python into Node.js/Electron applications introduces significant complexity:

| Challenge | Python Approach | This Package |
|-----------|-----------------|--------------|
| **Runtime dependency** | Requires Python interpreter + pip packages | Single `.node` binary + shared library |
| **Distribution** | Bundle Python or require user installation | npm install (prebuilt binaries) |
| **Electron/Node integration** | Child process or Python bridge | Native N-API module |
| **Cold start time** | Python interpreter startup overhead | Native code, instant loading |
| **Cross-platform packaging** | Complex (pyinstaller, conda, etc.) | Standard npm workflow |

**This package provides the same CTranslate2 performance with zero Python dependencies.**

---

## Installation

```bash
npm install faster-whisper-node
```

### Platform Requirements

| Platform | Additional Requirements |
|----------|------------------------|
| **Linux** | None |
| **Windows** | [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| **macOS** | None |

---

## API Reference

### Model Management

```javascript
const { 
  downloadModel,
  availableModels,
  isModelAvailable,
  getModelPath,
  getCacheDir
} = require('faster-whisper-node');
```

#### `downloadModel(size: string, cacheDir?: string): Promise<string>`

Download a model from HuggingFace Hub. Returns the path to the downloaded model.

```javascript
// Download to default cache (~/.cache/faster-whisper-node/models)
const path = await downloadModel('tiny');

// Download to custom directory
const path2 = await downloadModel('tiny', '/my/models');
```

#### `availableModels(): string[]`

Returns list of available model sizes.

```javascript
const models = availableModels();
// ["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
//  "medium", "medium.en", "large-v1", "large-v2", "large-v3", "distil-large-v3"]
```

#### `isModelAvailable(size: string): boolean`

Check if a model is downloaded in the cache.

```javascript
if (!isModelAvailable('tiny')) {
  await downloadModel('tiny');
}
```

#### `getModelPath(size: string): string`

Get the cache path for a model size.

```javascript
const path = getModelPath('tiny');
// "/Users/you/.cache/faster-whisper-node/models/tiny"
```

#### `getCacheDir(): string`

Get the default cache directory.

```javascript
const cacheDir = getCacheDir();
// "/Users/you/.cache/faster-whisper-node/models"
```

### Audio Decoding

```javascript
const { decodeAudio, decodeAudioBuffer } = require('faster-whisper-node');
```

#### `decodeAudio(path: string): number[]`

Decode any audio file to 16kHz mono samples.

```javascript
const samples = decodeAudio('./audio.mp3');
console.log(`${samples.length} samples = ${samples.length / 16000} seconds`);
```

#### `decodeAudioBuffer(buffer: Buffer): number[]`

Decode audio from a Buffer.

```javascript
const buffer = fs.readFileSync('./audio.mp3');
const samples = decodeAudioBuffer(buffer);
```

### Engine

```typescript
import { Engine, TranscribeOptions, ModelOptions, TranscriptionResult, Segment } from 'faster-whisper-node';
```

#### Constructor

```typescript
// From model name (uses cache directory)
const engine = new Engine('tiny');

// From explicit path
const engine = new Engine('./models/tiny');

// With options
const engine = Engine.withOptions('tiny', {
  device: 'cpu',
  computeType: 'int8',
  cpuThreads: 4
});
```

**ModelOptions:**
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `device` | `"cpu" \| "cuda"` | `"cpu"` | Computation device |
| `computeType` | `string` | `"default"` | `"default"`, `"auto"`, `"int8"`, `"int8_float16"`, `"float16"`, `"float32"` |
| `cpuThreads` | `number` | `0` | Number of CPU threads (0 = auto) |
| `cacheDir` | `string` | system cache | Custom cache directory |

#### Methods

##### `transcribe(audioPath: string): string`

Simple transcription, returns text only. Supports WAV, MP3, FLAC, OGG, M4A.

```typescript
const text = engine.transcribe('./audio.mp3');
// "Hello world."
```

##### `transcribeFile(audioPath: string, options?: TranscribeOptions): TranscriptionResult`

Full transcription with segments and metadata.

```typescript
const result = engine.transcribeFile('./audio.mp3', { language: 'en' });
console.log(result.text);      // "Hello world."
console.log(result.duration);  // 2.5
console.log(result.segments);  // [{ id: 0, start: 0, end: 2.5, text: "Hello world.", ... }]
```

##### `transcribeBuffer(buffer: Buffer, options?: TranscribeOptions): TranscriptionResult`

Transcribe from a Buffer containing audio data (any supported format).

```typescript
const buffer = fs.readFileSync('./audio.mp3');
const result = engine.transcribeBuffer(buffer);
```

##### `transcribeSamples(samples: number[], options?: TranscribeOptions): TranscriptionResult`

Transcribe from raw audio samples (must be 16kHz mono, normalized to [-1, 1]).

```typescript
const samples = decodeAudio('./audio.mp3');
const result = engine.transcribeSamples(samples);
```

##### `detectLanguage(audioPath: string): LanguageDetectionResult`

Detect the language of audio (requires multilingual model).

```typescript
const result = engine.detectLanguage('./audio.wav');
console.log(result.language);    // "en"
console.log(result.probability); // 0.95
```

##### `samplingRate(): number`

Returns expected sampling rate (16000 Hz).

##### `isMultilingual(): boolean`

Returns true if model supports multiple languages.

##### `numLanguages(): number`

Returns number of supported languages.

#### TranscribeOptions

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `language` | `string` | auto-detect | Source language code (e.g., "en", "de", "fr") |
| `task` | `"transcribe" \| "translate"` | `"transcribe"` | Task to perform |
| `beamSize` | `number` | `5` | Beam size (1 = greedy search) |
| `patience` | `number` | `1.0` | Beam search patience |
| `lengthPenalty` | `number` | `1.0` | Length penalty |
| `repetitionPenalty` | `number` | `1.0` | Repetition penalty (>1 to penalize) |
| `noRepeatNgramSize` | `number` | `0` | Prevent ngram repetitions |
| `temperature` | `number` | `1.0` | Sampling temperature |
| `suppressBlank` | `boolean` | `true` | Suppress blank outputs |
| `maxLength` | `number` | `448` | Maximum generation length |
| `wordTimestamps` | `boolean` | `false` | Include word-level timestamps |
| `initialPrompt` | `string` | - | Initial prompt for context |
| `prefix` | `string` | - | Prefix for first segment |
| `conditionOnPreviousText` | `boolean` | `true` | Condition on previous text |
| `compressionRatioThreshold` | `number` | `2.4` | Compression ratio threshold |
| `logProbThreshold` | `number` | `-1.0` | Log probability threshold |
| `noSpeechThreshold` | `number` | `0.6` | No speech probability threshold |
| `vadFilter` | `boolean` | `false` | Enable Voice Activity Detection |
| `vadOptions` | `VadOptions` | - | VAD configuration options |
| `hallucinationSilenceThreshold` | `number` | - | Skip segments with silence > threshold |

#### VadOptions

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `threshold` | `number` | `0.5` | Speech detection threshold (0.0 to 1.0) |
| `minSpeechDurationMs` | `number` | `250` | Minimum speech duration in ms |
| `maxSpeechDurationS` | `number` | `30` | Maximum speech duration in seconds |
| `minSilenceDurationMs` | `number` | `2000` | Minimum silence to split segments |
| `windowSizeMs` | `number` | `30` | Analysis window size in ms |
| `speechPadMs` | `number` | `400` | Padding around speech segments |

#### TranscriptionResult

```typescript
interface TranscriptionResult {
  segments: Segment[];
  language: string;
  languageProbability: number;
  duration: number;
  durationAfterVad: number;  // Duration after VAD filtering
  text: string;
}
```

#### Segment

```typescript
interface Segment {
  id: number;
  seek: number;
  start: number;  // seconds
  end: number;    // seconds
  text: string;
  tokens: number[];
  temperature: number;
  avgLogprob: number;
  compressionRatio: number;
  noSpeechProb: number;
  words?: Word[];  // Word-level timestamps (if enabled)
}
```

#### Word

```typescript
interface Word {
  word: string;
  start: number;      // seconds
  end: number;        // seconds
  probability: number;
}
```

#### LanguageDetectionResult

```typescript
interface LanguageDetectionResult {
  language: string;
  probability: number;
}
```

### Utility Functions

#### `formatTimestamp(seconds: number, alwaysIncludeHours?: boolean): string`

Format seconds to timestamp string.

```typescript
formatTimestamp(65.5);       // "01:05.500"
formatTimestamp(65.5, true); // "00:01:05.500"
formatTimestamp(3661.5);     // "01:01:01.500"
```

### GPU Detection

```javascript
const { isGpuAvailable, getGpuCount, getBestDevice } = require('faster-whisper-node');
```

#### `isGpuAvailable(): boolean`

Check if CUDA (GPU acceleration) is available.

```typescript
if (isGpuAvailable()) {
  console.log('GPU available, using CUDA');
}
```

#### `getGpuCount(): number`

Get the number of available CUDA devices.

```typescript
console.log(`Found ${getGpuCount()} GPUs`);
```

#### `getBestDevice(): string`

Get the best available device ("cuda" if GPU available, otherwise "cpu").

```typescript
const engine = Engine.withOptions('tiny', { device: getBestDevice() });
// Or simply use 'auto':
const engine2 = Engine.withOptions('tiny', { device: 'auto' });
```

---

## Supported Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | PCM 8/16/24/32-bit |
| MP3 | `.mp3` | Via symphonia |
| FLAC | `.flac` | Via symphonia |
| OGG Vorbis | `.ogg` | Via symphonia |
| AAC/M4A | `.m4a`, `.aac` | Via symphonia |

Audio is automatically:
- Converted to mono (if stereo)
- Resampled to 16kHz (if different sample rate)
- Normalized to [-1, 1] range

---

## Models

Models are downloaded automatically from HuggingFace Hub:

```javascript
// Download once (cached for future use)
await downloadModel('tiny');

// Or let the Engine download on first use
const engine = new Engine('tiny');  // Downloads if not cached
```

**Available models:**

| Model | Size | Speed | Quality | Multilingual |
|-------|------|-------|---------|--------------|
| `tiny` | 75 MB | ★★★★★ | ★ | ✅ |
| `tiny.en` | 75 MB | ★★★★★ | ★★ | ❌ |
| `base` | 142 MB | ★★★★ | ★★ | ✅ |
| `base.en` | 142 MB | ★★★★ | ★★★ | ❌ |
| `small` | 466 MB | ★★★ | ★★★ | ✅ |
| `small.en` | 466 MB | ★★★ | ★★★★ | ❌ |
| `medium` | 1.5 GB | ★★ | ★★★★ | ✅ |
| `medium.en` | 1.5 GB | ★★ | ★★★★★ | ❌ |
| `large-v2` | 3 GB | ★ | ★★★★★ | ✅ |
| `large-v3` | 3 GB | ★ | ★★★★★ | ✅ |
| `distil-large-v3` | 1.5 GB | ★★★ | ★★★★★ | ✅ |

**Manual download:**

```bash
mkdir -p models && cd models
git lfs install
git clone --depth 1 https://huggingface.co/Systran/faster-whisper-tiny tiny
```

---

## Platform Support

| Platform | Architecture | Status |
|----------|--------------|--------|
| Linux | x86_64, arm64 | ✅ |
| macOS | arm64 (Apple Silicon), x86_64 | ✅ |
| Windows | x86_64 | ✅ |

---

## Development

```bash
# Clone
git clone --recursive https://github.com/manmal/faster-whisper-node
cd faster-whisper-node

# Install dependencies (downloads prebuilt CTranslate2)
npm install

# Build
npm run build

# Test
npm test
```

---

## Credits

- **[SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — Original implementation and model conversions
- **[OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2)** — The inference engine
- **[ct2rs](https://github.com/jkawamoto/ctranslate2-rs)** — Rust bindings for CTranslate2
- **[symphonia](https://github.com/pdeljanov/Symphonia)** — Pure Rust audio decoding

---

## License

MIT — See [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md) for third-party licenses.
