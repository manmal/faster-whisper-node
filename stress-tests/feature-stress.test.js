#!/usr/bin/env node
/**
 * Feature Stress Test
 * 
 * Tests all API features under load to ensure they work correctly
 * when called many times in succession.
 * 
 * Run with: node stress-tests/feature-stress.test.js [options]
 * 
 * Options:
 *   --iterations=N   Iterations per feature (default: 20)
 *   --verbose        Show detailed output
 */

const path = require('path');
const fs = require('fs');

// Parse command line args
const args = {};
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=');
  args[key] = value || true;
}

const ITERATIONS = parseInt(args.iterations || '20');
const VERBOSE = args.verbose === true;
const MODEL_PATH = args.model || './models/tiny';

// Check model exists
if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
  console.error('❌ Model not found:', MODEL_PATH);
  console.error('   Download with: cd models && git lfs install && git clone --depth 1 https://huggingface.co/Systran/faster-whisper-tiny tiny');
  process.exit(1);
}

const {
  Engine,
  decodeAudio,
  decodeAudioBuffer,
  formatTimestamp,
  availableModels,
  isModelAvailable,
  getModelPath,
  getCacheDir,
  isGpuAvailable,
  getGpuCount,
  getBestDevice,
} = require('../index');

const AUDIO_FILES = [
  './tests/fixtures/hello.wav',
  './tests/fixtures/numbers.wav',
  './tests/fixtures/sentence.wav',
].filter(f => fs.existsSync(f));

if (AUDIO_FILES.length === 0) {
  console.error('❌ No test audio files found in tests/fixtures/');
  process.exit(1);
}

const results = {
  passed: 0,
  failed: 0,
  errors: [],
};

function log(msg) {
  if (VERBOSE) console.log(msg);
}

function test(name, fn) {
  const start = Date.now();
  let success = 0;
  let fail = 0;
  const errors = [];
  
  for (let i = 0; i < ITERATIONS; i++) {
    try {
      fn(i);
      success++;
    } catch (err) {
      fail++;
      if (errors.length < 3) {
        errors.push(err.message);
      }
    }
  }
  
  const elapsed = Date.now() - start;
  const status = fail === 0 ? '✅' : '❌';
  console.log(`${status} ${name}: ${success}/${ITERATIONS} passed (${elapsed}ms)`);
  
  if (fail > 0) {
    results.failed++;
    results.errors.push({ name, errors: errors.slice(0, 3) });
  } else {
    results.passed++;
  }
  
  return fail === 0;
}

async function runTests() {
  console.log('='.repeat(60));
  console.log('Feature Stress Test');
  console.log('='.repeat(60));
  console.log(`Iterations per test: ${ITERATIONS}`);
  console.log(`Model:               ${MODEL_PATH}`);
  console.log(`Audio files:         ${AUDIO_FILES.length}`);
  console.log('');
  
  // Load audio buffers
  const audioBuffers = AUDIO_FILES.map(f => fs.readFileSync(f));
  
  console.log('Creating engine...\n');
  const engine = new Engine(MODEL_PATH);
  
  console.log('=== Utility Functions ===\n');
  
  test('formatTimestamp', (i) => {
    const ts = formatTimestamp(i * 0.5 + 0.123);
    if (typeof ts !== 'string' || !ts.includes(':')) {
      throw new Error(`Invalid timestamp: ${ts}`);
    }
  });
  
  test('availableModels', () => {
    const models = availableModels();
    if (!Array.isArray(models) || models.length === 0) {
      throw new Error('No models returned');
    }
  });
  
  test('isModelAvailable', () => {
    const result = isModelAvailable('tiny');
    // Result depends on whether model is cached
    if (typeof result !== 'boolean') {
      throw new Error(`Expected boolean, got ${typeof result}`);
    }
  });
  
  test('getModelPath', () => {
    const path = getModelPath('tiny');
    if (typeof path !== 'string' || path.length === 0) {
      throw new Error(`Invalid path: ${path}`);
    }
  });
  
  test('getCacheDir', () => {
    const dir = getCacheDir();
    if (typeof dir !== 'string' || dir.length === 0) {
      throw new Error(`Invalid cache dir: ${dir}`);
    }
  });
  
  test('isGpuAvailable', () => {
    const result = isGpuAvailable();
    if (typeof result !== 'boolean') {
      throw new Error(`Expected boolean, got ${typeof result}`);
    }
  });
  
  test('getGpuCount', () => {
    const count = getGpuCount();
    if (typeof count !== 'number' || count < 0) {
      throw new Error(`Invalid GPU count: ${count}`);
    }
  });
  
  test('getBestDevice', () => {
    const device = getBestDevice();
    if (device !== 'cpu' && device !== 'cuda') {
      throw new Error(`Invalid device: ${device}`);
    }
  });
  
  console.log('\n=== Audio Decoding ===\n');
  
  test('decodeAudio', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const samples = decodeAudio(audioFile);
    if (!Array.isArray(samples) || samples.length === 0) {
      throw new Error('No samples decoded');
    }
    // Verify samples are normalized
    const max = Math.max(...samples.slice(0, 1000).map(Math.abs));
    if (max > 1.1) {
      throw new Error(`Samples not normalized: max=${max}`);
    }
  });
  
  test('decodeAudioBuffer', (i) => {
    const buffer = audioBuffers[i % audioBuffers.length];
    const samples = decodeAudioBuffer(buffer);
    if (!Array.isArray(samples) || samples.length === 0) {
      throw new Error('No samples decoded');
    }
  });
  
  console.log('\n=== Engine Properties ===\n');
  
  test('samplingRate', () => {
    const rate = engine.samplingRate();
    if (rate !== 16000) {
      throw new Error(`Expected 16000, got ${rate}`);
    }
  });
  
  test('isMultilingual', () => {
    const result = engine.isMultilingual();
    if (typeof result !== 'boolean') {
      throw new Error(`Expected boolean, got ${typeof result}`);
    }
  });
  
  test('numLanguages', () => {
    const count = engine.numLanguages();
    if (typeof count !== 'number' || count < 1) {
      throw new Error(`Invalid language count: ${count}`);
    }
  });
  
  console.log('\n=== Transcription Methods ===\n');
  
  test('transcribe (legacy)', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const text = engine.transcribe(audioFile);
    if (typeof text !== 'string' || text.trim().length === 0) {
      throw new Error('Empty transcription');
    }
  });
  
  test('transcribeFile', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const result = engine.transcribeFile(audioFile);
    if (!result.text || !result.segments) {
      throw new Error('Invalid result structure');
    }
  });
  
  test('transcribeFile with options', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const result = engine.transcribeFile(audioFile, {
      language: 'en',
      beamSize: 3,
    });
    if (!result.text || result.language !== 'en') {
      throw new Error('Options not applied');
    }
  });
  
  test('transcribeBuffer', (i) => {
    const buffer = audioBuffers[i % audioBuffers.length];
    const result = engine.transcribeBuffer(buffer);
    if (!result.text || !result.segments) {
      throw new Error('Invalid result structure');
    }
  });
  
  test('transcribeSamples', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const samples = decodeAudio(audioFile);
    const result = engine.transcribeSamples(samples);
    if (!result.text || !result.segments) {
      throw new Error('Invalid result structure');
    }
  });
  
  test('transcribeWithOptions', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const text = engine.transcribeWithOptions(audioFile, { language: 'en' });
    if (typeof text !== 'string') {
      throw new Error('Expected string result');
    }
  });
  
  console.log('\n=== Word Timestamps ===\n');
  
  test('transcribeFile with wordTimestamps', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const result = engine.transcribeFile(audioFile, {
      wordTimestamps: true,
    });
    if (!result.segments || result.segments.length === 0) {
      throw new Error('No segments returned');
    }
    // Words may or may not be present depending on model
    // Just check the structure is valid
  });
  
  console.log('\n=== Language Detection ===\n');
  
  test('detectLanguage', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const result = engine.detectLanguage(audioFile);
    if (!result.language || typeof result.probability !== 'number') {
      throw new Error('Invalid detection result');
    }
  });
  
  test('detectLanguageBuffer', (i) => {
    const buffer = audioBuffers[i % audioBuffers.length];
    const result = engine.detectLanguageBuffer(buffer);
    if (!result.language || typeof result.probability !== 'number') {
      throw new Error('Invalid detection result');
    }
  });
  
  console.log('\n=== VAD (Voice Activity Detection) ===\n');
  
  test('transcribeFile with VAD', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const result = engine.transcribeFile(audioFile, {
      vadFilter: true,
    });
    if (!result.text) {
      throw new Error('No text returned');
    }
    // durationAfterVad should be set
    if (typeof result.durationAfterVad !== 'number') {
      throw new Error('durationAfterVad not set');
    }
  });
  
  test('transcribeFile with VAD options', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const result = engine.transcribeFile(audioFile, {
      vadFilter: true,
      vadOptions: {
        threshold: 0.5,
        minSpeechDurationMs: 100,
        minSilenceDurationMs: 500,
      },
    });
    if (!result.text) {
      throw new Error('No text returned');
    }
  });
  
  console.log('\n=== Result Structure ===\n');
  
  test('segment structure', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const result = engine.transcribeFile(audioFile);
    for (const seg of result.segments) {
      if (typeof seg.id !== 'number') throw new Error('Missing segment id');
      if (typeof seg.start !== 'number') throw new Error('Missing segment start');
      if (typeof seg.end !== 'number') throw new Error('Missing segment end');
      if (typeof seg.text !== 'string') throw new Error('Missing segment text');
      if (!Array.isArray(seg.tokens)) throw new Error('Missing tokens array');
      if (typeof seg.temperature !== 'number') throw new Error('Missing temperature');
      if (typeof seg.avgLogprob !== 'number') throw new Error('Missing avgLogprob');
      if (typeof seg.noSpeechProb !== 'number') throw new Error('Missing noSpeechProb');
    }
  });
  
  test('result structure', (i) => {
    const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
    const result = engine.transcribeFile(audioFile);
    if (typeof result.language !== 'string') throw new Error('Missing language');
    if (typeof result.languageProbability !== 'number') throw new Error('Missing languageProbability');
    if (typeof result.duration !== 'number') throw new Error('Missing duration');
    if (typeof result.durationAfterVad !== 'number') throw new Error('Missing durationAfterVad');
    if (typeof result.text !== 'string') throw new Error('Missing text');
  });
  
  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('Summary');
  console.log('='.repeat(60));
  console.log(`Total tests:     ${results.passed + results.failed}`);
  console.log(`Passed:          ${results.passed}`);
  console.log(`Failed:          ${results.failed}`);
  console.log(`Pass rate:       ${(results.passed / (results.passed + results.failed) * 100).toFixed(1)}%`);
  
  if (results.errors.length > 0) {
    console.log('\nFailed tests:');
    for (const { name, errors } of results.errors) {
      console.log(`  ${name}:`);
      for (const err of errors) {
        console.log(`    - ${err}`);
      }
    }
  }
  
  console.log('\n' + (results.failed === 0 ? '✅ All tests passed!' : '❌ Some tests failed'));
  
  if (results.failed > 0) {
    process.exit(1);
  }
}

runTests().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
