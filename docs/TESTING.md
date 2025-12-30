# Testing Guide

This guide covers the testing infrastructure for faster-whisper-rs.

## Test Structure

```
tests/
├── unit/           # Unit tests for individual functions
│   ├── engine.test.js
│   ├── utils.test.js
│   └── audio.test.js
├── integration/    # Integration tests for combined functionality
│   ├── transcription.test.js
│   └── engine-options.test.js
├── e2e/           # End-to-end transcription tests
│   └── transcription.test.js
├── fixtures/      # Test audio files
│   ├── hello.wav
│   ├── hello.mp3
│   ├── hello.flac
│   ├── hello.ogg
│   ├── numbers.wav
│   └── sentence.wav
├── phase3.test.js # Phase 3 feature tests (word timestamps, VAD)
└── phase4.test.js # Phase 4 feature tests (GPU detection)
```

## Running Tests

### Quick Tests
```bash
# Basic functionality test (fast)
npm test
```

### Full Test Suite
```bash
# All unit, integration, and e2e tests
npm run test:all

# Comprehensive: includes all tests + phase tests
npm run test:comprehensive
```

### Individual Test Types
```bash
# Unit tests only
npm run test:unit

# Integration tests only
npm run test:integration

# E2E tests only
npm run test:e2e

# Phase 3 features (word timestamps, VAD, hallucination detection)
npm run test:phase3

# Phase 4 features (GPU detection, device selection)
npm run test:phase4
```

## Test Requirements

### Model
Tests require the `tiny` model. Download with:
```bash
cd models
git lfs install
git clone --depth 1 https://huggingface.co/Systran/faster-whisper-tiny tiny
```

Or use the auto-download feature:
```javascript
const { downloadModel } = require('faster-whisper-rs');
await downloadModel('tiny');
```

### Test Audio Files
Test fixtures are included in `tests/fixtures/`:
- `hello.wav` - ~0.6s "Hello world"
- `numbers.wav` - ~2.8s counting numbers
- `sentence.wav` - ~4.4s longer sentence
- `hello.mp3`, `hello.flac`, `hello.ogg` - Same audio in different formats

## Benchmarks

Run performance benchmarks:
```bash
npm run benchmark
```

This measures:
- Audio decoding performance
- Transcription speed for different file lengths
- Input methods (file, buffer, samples)
- Transcription options overhead
- Engine creation time
- Real-time factor (RTF)

### Sample Output
```
Real-Time Factor (RTF)
------------------------------------------------------------
  (RTF < 1 means faster than real-time)

  hello.wav (0.63s):    RTF = 0.872
  numbers.wav (1.39s): RTF = 0.419
  sentence.wav (2.19s): RTF = 0.270
```

## Writing Tests

### Using Node.js Built-in Test Runner

Tests use Node.js built-in test runner (Node 18+):

```javascript
const { describe, it, before } = require('node:test');
const assert = require('node:assert');

describe('Feature', () => {
  it('should work', () => {
    assert.strictEqual(1 + 1, 2);
  });
});
```

### Conditional Tests

Skip tests when prerequisites aren't met:

```javascript
const hasModel = fs.existsSync(path.join(modelPath, 'model.bin'));

describe('Transcription', { skip: !hasModel && 'Model not available' }, () => {
  // Tests here
});
```

### Assertions

Use Node.js assert module:

```javascript
// Strict equality
assert.strictEqual(result, expected);

// Truthiness
assert.ok(condition, 'Error message');

// Throws
assert.throws(() => {
  throw new Error('test');
}, /Error/);

// Deep equality
assert.deepStrictEqual(obj1, obj2);
```

## CI Integration

Tests run automatically on GitHub Actions:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: npm run test:all
```

## Coverage

Currently no coverage tool is configured. To add coverage:

```bash
npm install -D c8
npx c8 npm run test:all
```
