#!/usr/bin/env node
/**
 * Transcription Performance Benchmarks
 * 
 * Run with: node benchmarks/transcription.bench.js
 */
const path = require('path');
const fs = require('fs');

const modelPath = './models/tiny';
const fixturesDir = './tests/fixtures';

// Check prerequisites
if (!fs.existsSync(path.join(modelPath, 'model.bin'))) {
  console.log('⚠️  Model not found. Download with:');
  console.log('   cd models && git lfs install && git clone --depth 1 https://huggingface.co/Systran/faster-whisper-tiny tiny');
  process.exit(0);
}

const { 
  Engine, 
  decodeAudio, 
  decodeAudioBuffer,
  isGpuAvailable,
  getBestDevice
} = require('../index');

// Utility for running benchmarks
function benchmark(name, fn, iterations = 10) {
  // Warmup
  for (let i = 0; i < 3; i++) {
    fn();
  }
  
  // Measure
  const times = [];
  for (let i = 0; i < iterations; i++) {
    const start = process.hrtime.bigint();
    fn();
    const end = process.hrtime.bigint();
    times.push(Number(end - start) / 1e6); // Convert to ms
  }
  
  // Calculate stats
  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const min = times[0];
  const max = times[times.length - 1];
  const stddev = Math.sqrt(times.reduce((sum, t) => sum + Math.pow(t - mean, 2), 0) / times.length);
  
  console.log(`  ${name}:`);
  console.log(`    Iterations: ${iterations}`);
  console.log(`    Mean:   ${mean.toFixed(2)} ms`);
  console.log(`    Median: ${median.toFixed(2)} ms`);
  console.log(`    Min:    ${min.toFixed(2)} ms`);
  console.log(`    Max:    ${max.toFixed(2)} ms`);
  console.log(`    StdDev: ${stddev.toFixed(2)} ms`);
  
  return { name, median, mean, min, max, stddev, iterations };
}

// Print system info
console.log('='.repeat(60));
console.log('faster-whisper-rs Benchmark Suite');
console.log('='.repeat(60));
console.log('');
console.log('System Info:');
console.log(`  Platform: ${process.platform}`);
console.log(`  Arch:     ${process.arch}`);
console.log(`  Node:     ${process.version}`);
console.log(`  GPU:      ${isGpuAvailable() ? 'Available' : 'Not available'}`);
console.log(`  Device:   ${getBestDevice()}`);
console.log('');

// Load engine (one-time setup)
console.log('Loading model...');
const engineStart = process.hrtime.bigint();
const engine = new Engine(modelPath);
const engineEnd = process.hrtime.bigint();
console.log(`  Model loaded in ${(Number(engineEnd - engineStart) / 1e6).toFixed(2)} ms`);
console.log('');

const results = [];

// Audio file paths
const helloWav = path.join(fixturesDir, 'hello.wav');
const helloMp3 = path.join(fixturesDir, 'hello.mp3');
const numbersWav = path.join(fixturesDir, 'numbers.wav');
const sentenceWav = path.join(fixturesDir, 'sentence.wav');

// Benchmark 1: Audio Decoding
console.log('-'.repeat(60));
console.log('Audio Decoding');
console.log('-'.repeat(60));

results.push(benchmark('decodeAudio (WAV)', () => {
  decodeAudio(helloWav);
}, 20));

results.push(benchmark('decodeAudio (MP3)', () => {
  decodeAudio(helloMp3);
}, 20));

const wavBuffer = fs.readFileSync(helloWav);
results.push(benchmark('decodeAudioBuffer (WAV)', () => {
  decodeAudioBuffer(wavBuffer);
}, 20));

console.log('');

// Benchmark 2: Basic Transcription
console.log('-'.repeat(60));
console.log('Basic Transcription');
console.log('-'.repeat(60));

results.push(benchmark('transcribe (short ~0.6s)', () => {
  engine.transcribe(helloWav);
}, 10));

results.push(benchmark('transcribe (numbers ~2.8s)', () => {
  engine.transcribe(numbersWav);
}, 10));

results.push(benchmark('transcribe (sentence ~4.4s)', () => {
  engine.transcribe(sentenceWav);
}, 10));

console.log('');

// Benchmark 3: Transcription Methods
console.log('-'.repeat(60));
console.log('Transcription Methods');
console.log('-'.repeat(60));

results.push(benchmark('transcribeFile (full result)', () => {
  engine.transcribeFile(helloWav);
}, 10));

results.push(benchmark('transcribeBuffer', () => {
  engine.transcribeBuffer(wavBuffer);
}, 10));

const samples = decodeAudio(helloWav);
results.push(benchmark('transcribeSamples', () => {
  engine.transcribeSamples(samples);
}, 10));

console.log('');

// Benchmark 4: Transcription Options
console.log('-'.repeat(60));
console.log('Transcription Options');
console.log('-'.repeat(60));

results.push(benchmark('wordTimestamps: true', () => {
  engine.transcribeFile(helloWav, { wordTimestamps: true });
}, 10));

results.push(benchmark('vadFilter: true', () => {
  engine.transcribeFile(helloWav, { vadFilter: true });
}, 10));

results.push(benchmark('beamSize: 1 (greedy)', () => {
  engine.transcribeFile(helloWav, { beamSize: 1 });
}, 10));

results.push(benchmark('beamSize: 5 (default)', () => {
  engine.transcribeFile(helloWav, { beamSize: 5 });
}, 10));

results.push(benchmark('language: "en" (skip detection)', () => {
  engine.transcribeFile(helloWav, { language: 'en' });
}, 10));

console.log('');

// Benchmark 5: Combined Options
console.log('-'.repeat(60));
console.log('Combined Options');
console.log('-'.repeat(60));

results.push(benchmark('VAD + Word timestamps', () => {
  engine.transcribeFile(helloWav, { 
    vadFilter: true,
    wordTimestamps: true 
  });
}, 10));

results.push(benchmark('Full production config', () => {
  engine.transcribeFile(helloWav, {
    language: 'en',
    beamSize: 5,
    temperature: 0.0,
    vadFilter: true,
    wordTimestamps: true,
    suppressBlank: true
  });
}, 10));

console.log('');

// Benchmark 6: Engine Creation
console.log('-'.repeat(60));
console.log('Engine Creation');
console.log('-'.repeat(60));

results.push(benchmark('new Engine()', () => {
  new Engine(modelPath);
}, 5));

results.push(benchmark('Engine.withOptions()', () => {
  Engine.withOptions(modelPath, { device: 'cpu', cpuThreads: 4 });
}, 5));

console.log('');

// Real-time factor calculation
console.log('-'.repeat(60));
console.log('Real-Time Factor (RTF)');
console.log('-'.repeat(60));

// Get audio durations
const helloDuration = decodeAudio(helloWav).length / 16000;
const numbersDuration = decodeAudio(numbersWav).length / 16000;
const sentenceDuration = decodeAudio(sentenceWav).length / 16000;

console.log('  (RTF < 1 means faster than real-time)');
console.log('');

// Run multiple times for RTF
const rtfIterations = 5;
let helloTotal = 0, numbersTotal = 0, sentenceTotal = 0;

for (let i = 0; i < rtfIterations; i++) {
  let start = process.hrtime.bigint();
  engine.transcribe(helloWav);
  helloTotal += Number(process.hrtime.bigint() - start) / 1e9;
  
  start = process.hrtime.bigint();
  engine.transcribe(numbersWav);
  numbersTotal += Number(process.hrtime.bigint() - start) / 1e9;
  
  start = process.hrtime.bigint();
  engine.transcribe(sentenceWav);
  sentenceTotal += Number(process.hrtime.bigint() - start) / 1e9;
}

const helloRtf = (helloTotal / rtfIterations) / helloDuration;
const numbersRtf = (numbersTotal / rtfIterations) / numbersDuration;
const sentenceRtf = (sentenceTotal / rtfIterations) / sentenceDuration;

console.log(`  hello.wav (${helloDuration.toFixed(2)}s):    RTF = ${helloRtf.toFixed(3)}`);
console.log(`  numbers.wav (${numbersDuration.toFixed(2)}s): RTF = ${numbersRtf.toFixed(3)}`);
console.log(`  sentence.wav (${sentenceDuration.toFixed(2)}s): RTF = ${sentenceRtf.toFixed(3)}`);
console.log('');

// Summary
console.log('='.repeat(60));
console.log('Summary');
console.log('='.repeat(60));
console.log('');
console.log('Top 5 fastest operations (by median):');
const sorted = [...results].sort((a, b) => a.median - b.median);
for (let i = 0; i < Math.min(5, sorted.length); i++) {
  console.log(`  ${i + 1}. ${sorted[i].name}: ${sorted[i].median.toFixed(2)} ms`);
}
console.log('');
console.log('Done!');
