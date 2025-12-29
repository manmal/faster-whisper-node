#!/usr/bin/env node
/**
 * Streaming Stress Test with Comprehensive Analysis
 * 
 * Simulates 100 concurrent "streaming" transcription sessions.
 * 
 * Since the API doesn't support true incremental streaming, we simulate it by:
 * - Processing audio in fixed-size windows (not cumulative)
 * - Each "chunk" represents a new segment of audio arriving
 * - We measure latency as: time_to_process_chunk - chunk_duration
 * 
 * For real-time: processing_time < chunk_duration
 * 
 * Run with: node stress-tests/streaming-analysis.test.js [options]
 */

const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const os = require('os');
const path = require('path');
const fs = require('fs');

// Parse command line args
const args = {};
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=');
  args[key] = value || true;
}

const CONCURRENT_STREAMS = parseInt(args.streams || '100');
const STREAM_DURATION_S = parseFloat(args.duration || '10'); // Total stream duration
const CHUNK_SIZE_MS = parseInt(args.chunk || '2000'); // Process 2s chunks (reasonable for Whisper)
const WORKER_COUNT = parseInt(args.workers || Math.min(os.cpus().length, 8));
const MODEL_PATH = args.model || './models/tiny';
const VERBOSE = args.verbose === true;

const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SIZE_MS / 1000);

if (!isMainThread) {
  runWorker();
} else {
  main().catch(console.error);
}

function runWorker() {
  const { Engine, decodeAudio } = require(workerData.enginePath);
  const engine = new Engine(workerData.modelPath);
  
  // Load and prepare audio samples
  const audioFiles = workerData.audioFiles;
  const allSamples = [];
  for (const f of audioFiles) {
    allSamples.push(...decodeAudio(f));
  }
  
  // Create extended audio by repeating
  const targetSamples = Math.ceil(workerData.streamDuration * SAMPLE_RATE);
  const extendedSamples = [];
  while (extendedSamples.length < targetSamples) {
    extendedSamples.push(...allSamples);
  }
  const streamSamples = new Float32Array(extendedSamples.slice(0, targetSamples));
  
  parentPort.on('message', (msg) => {
    if (msg.type === 'process_stream') {
      processStream(engine, streamSamples, msg.streamId);
    } else if (msg.type === 'exit') {
      process.exit(0);
    }
  });
  
  parentPort.postMessage({ type: 'ready' });
  
  function processStream(engine, samples, streamId) {
    const results = {
      streamId,
      chunks: [],
      words: [],
      errors: [],
      startTime: Date.now(),
    };
    
    const chunkDurationMs = workerData.chunkSizeMs;
    const chunkSamples = workerData.chunkSamples;
    const totalChunks = Math.ceil(samples.length / chunkSamples);
    
    // Process each chunk independently (simulating sliding window / VAD segments)
    for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
      const chunkStartSample = chunkIdx * chunkSamples;
      const chunkEndSample = Math.min((chunkIdx + 1) * chunkSamples, samples.length);
      const chunkAudio = samples.slice(chunkStartSample, chunkEndSample);
      
      const audioStartS = chunkStartSample / SAMPLE_RATE;
      const audioEndS = chunkEndSample / SAMPLE_RATE;
      const actualChunkDurationMs = (chunkEndSample - chunkStartSample) / SAMPLE_RATE * 1000;
      
      const processStart = Date.now();
      
      try {
        const result = engine.transcribeSamples(Array.from(chunkAudio), {
          language: 'en',
          wordTimestamps: false, // wordTimestamps is very slow, disable for real-time
          beamSize: 5, // beam=5 is actually faster than greedy for tiny model
        });
        
        const processEnd = Date.now();
        const processingTimeMs = processEnd - processStart;
        const wallClockSinceStart = processEnd - results.startTime;
        
        // Latency relative to real-time audio position
        // If we're processing chunk ending at audioEndS, and it's now wallClockSinceStart ms since stream start
        // Real-time would mean wallClockSinceStart <= audioEndS * 1000
        const latencyVsRealtimeMs = wallClockSinceStart - (audioEndS * 1000);
        
        // Is this chunk processed faster than real-time?
        const realTimeFactor = processingTimeMs / actualChunkDurationMs;
        
        results.chunks.push({
          chunkIdx,
          audioStartS,
          audioEndS,
          audioDurationMs: actualChunkDurationMs,
          processingTimeMs,
          wallClockSinceStartMs: wallClockSinceStart,
          latencyVsRealtimeMs,
          realTimeFactor,
          segmentCount: result.segments.length,
          text: result.text.trim().substring(0, 50),
        });
        
        // Extract words with timing (adjusted to absolute audio time)
        for (const seg of result.segments) {
          if (seg.words) {
            for (const word of seg.words) {
              results.words.push({
                word: word.word.trim(),
                // Absolute audio time = chunk start + relative word time
                audioStartS: audioStartS + word.start,
                audioEndS: audioStartS + word.end,
                wallClockMs: wallClockSinceStart,
                // Word latency: wall clock time minus when the word ended in audio
                latencyMs: wallClockSinceStart - ((audioStartS + word.end) * 1000),
                probability: word.probability,
                chunkIdx,
              });
            }
          }
        }
        
      } catch (err) {
        results.errors.push({
          chunkIdx,
          error: err.message,
        });
      }
    }
    
    results.endTime = Date.now();
    results.totalTimeMs = results.endTime - results.startTime;
    results.audioDurationS = samples.length / SAMPLE_RATE;
    results.overallRTF = results.totalTimeMs / (results.audioDurationS * 1000);
    
    parentPort.postMessage({ type: 'stream_complete', results });
  }
}

async function main() {
  console.log('═'.repeat(70));
  console.log('  STREAMING STRESS TEST - 100 CONCURRENT STREAMS');
  console.log('═'.repeat(70));
  console.log(`  Concurrent streams:  ${CONCURRENT_STREAMS}`);
  console.log(`  Stream duration:     ${STREAM_DURATION_S}s`);
  console.log(`  Chunk size:          ${CHUNK_SIZE_MS}ms`);
  console.log(`  Worker threads:      ${WORKER_COUNT}`);
  console.log(`  Model:               ${MODEL_PATH}`);
  console.log('═'.repeat(70));
  
  // Verify prerequisites
  if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
    console.error('❌ Model not found:', MODEL_PATH);
    process.exit(1);
  }
  
  const audioFiles = [
    './tests/fixtures/hello.wav',
    './tests/fixtures/numbers.wav', 
    './tests/fixtures/sentence.wav',
  ].filter(f => fs.existsSync(f));
  
  console.log(`\n📁 Audio: ${audioFiles.length} files, extended to ${STREAM_DURATION_S}s each`);
  
  // Initialize workers
  console.log(`\n🔧 Initializing ${WORKER_COUNT} workers...`);
  
  const workers = [];
  const allResults = [];
  let completedStreams = 0;
  let nextStreamId = 0;
  
  const enginePath = path.resolve(__dirname, '../index.js');
  const modelPathAbs = path.resolve(MODEL_PATH);
  const audioFilesAbs = audioFiles.map(f => path.resolve(f));
  
  await new Promise((resolve, reject) => {
    let readyWorkers = 0;
    
    for (let i = 0; i < WORKER_COUNT; i++) {
      const worker = new Worker(__filename, {
        workerData: {
          enginePath,
          modelPath: modelPathAbs,
          audioFiles: audioFilesAbs,
          streamDuration: STREAM_DURATION_S,
          chunkSamples: CHUNK_SAMPLES,
          chunkSizeMs: CHUNK_SIZE_MS,
        },
      });
      
      worker.on('message', (msg) => {
        if (msg.type === 'ready') {
          readyWorkers++;
          if (readyWorkers === WORKER_COUNT) {
            console.log(`  All ${WORKER_COUNT} workers ready`);
            resolve();
          }
        } else if (msg.type === 'stream_complete') {
          allResults.push(msg.results);
          completedStreams++;
          
          const pct = (completedStreams / CONCURRENT_STREAMS * 100).toFixed(0);
          process.stdout.write(`\r  Progress: ${completedStreams}/${CONCURRENT_STREAMS} (${pct}%)`);
          
          if (nextStreamId < CONCURRENT_STREAMS) {
            worker.postMessage({ type: 'process_stream', streamId: nextStreamId++ });
          }
        }
      });
      
      worker.on('error', (err) => {
        console.error(`\nWorker ${i} error:`, err);
      });
      
      workers.push(worker);
    }
  });
  
  console.log(`\n🚀 Starting ${CONCURRENT_STREAMS} streams...\n`);
  const testStartTime = Date.now();
  
  // Initial dispatch
  for (let i = 0; i < Math.min(WORKER_COUNT, CONCURRENT_STREAMS); i++) {
    workers[i].postMessage({ type: 'process_stream', streamId: nextStreamId++ });
  }
  
  // Wait for completion
  await new Promise((resolve) => {
    const check = setInterval(() => {
      if (completedStreams >= CONCURRENT_STREAMS) {
        clearInterval(check);
        resolve();
      }
    }, 100);
  });
  
  const totalTestTime = Date.now() - testStartTime;
  console.log('\n\n');
  
  // Terminate workers
  for (const worker of workers) {
    worker.postMessage({ type: 'exit' });
  }
  
  analyzeResults(allResults, totalTestTime);
}

function analyzeResults(results, totalTestTime) {
  console.log('═'.repeat(70));
  console.log('  COMPREHENSIVE STATISTICAL ANALYSIS');
  console.log('═'.repeat(70));
  
  const allChunks = results.flatMap(r => r.chunks);
  const allWords = results.flatMap(r => r.words);
  const allErrors = results.flatMap(r => r.errors);
  
  // ============== OVERVIEW ==============
  console.log('\n📊 OVERVIEW');
  console.log('─'.repeat(50));
  console.log(`  Total streams:           ${results.length}`);
  console.log(`  Total test time:         ${(totalTestTime / 1000).toFixed(2)}s`);
  console.log(`  Total chunks:            ${allChunks.length}`);
  console.log(`  Total words detected:    ${allWords.length}`);
  console.log(`  Total errors:            ${allErrors.length}`);
  console.log(`  Success rate:            ${((results.length - results.filter(r => r.errors.length > 0).length) / results.length * 100).toFixed(1)}%`);
  
  // ============== OVERALL REAL-TIME FACTOR ==============
  console.log('\n⏱️  OVERALL REAL-TIME FACTOR (per stream)');
  console.log('─'.repeat(50));
  console.log('  RTF < 1.0 means faster than real-time (good)');
  console.log('  RTF > 1.0 means slower than real-time (bad)');
  
  const overallRTFs = results.map(r => r.overallRTF);
  printDetailedStats('Overall RTF', overallRTFs, 'x', 3);
  
  const fasterCount = overallRTFs.filter(r => r < 1.0).length;
  console.log(`\n  ✅ Faster than real-time: ${fasterCount}/${results.length} (${(fasterCount/results.length*100).toFixed(1)}%)`);
  console.log(`  ❌ Slower than real-time: ${results.length - fasterCount}/${results.length}`);
  
  // ============== PER-CHUNK REAL-TIME FACTOR ==============
  console.log('\n⚡ PER-CHUNK REAL-TIME FACTOR');
  console.log('─'.repeat(50));
  console.log(`  Each chunk is ${CHUNK_SIZE_MS}ms of audio`);
  
  const chunkRTFs = allChunks.map(c => c.realTimeFactor);
  printDetailedStats('Chunk RTF', chunkRTFs, 'x', 3);
  
  const fastChunks = chunkRTFs.filter(r => r < 1.0).length;
  console.log(`\n  Chunks faster than real-time: ${fastChunks}/${allChunks.length} (${(fastChunks/allChunks.length*100).toFixed(1)}%)`);
  
  // ============== CHUNK PROCESSING TIME ==============
  console.log('\n🔧 CHUNK PROCESSING TIME');
  console.log('─'.repeat(50));
  
  const processingTimes = allChunks.map(c => c.processingTimeMs);
  printDetailedStats('Processing time', processingTimes, 'ms', 1);
  
  // ============== END-TO-END LATENCY ==============
  console.log('\n📈 END-TO-END LATENCY (vs real-time audio position)');
  console.log('─'.repeat(50));
  console.log('  Latency = wall_clock_time - audio_position');
  console.log('  Negative = ahead of real-time, Positive = behind');
  
  const latencies = allChunks.map(c => c.latencyVsRealtimeMs);
  printDetailedStats('Latency', latencies, 'ms', 1);
  
  // By position in stream
  const chunks = results.length > 0 ? results[0].chunks.length : 0;
  if (chunks > 0) {
    console.log('\n  Latency by chunk position (averaged across streams):');
    for (let i = 0; i < Math.min(chunks, 10); i++) {
      const chunkLatencies = allChunks.filter(c => c.chunkIdx === i).map(c => c.latencyVsRealtimeMs);
      if (chunkLatencies.length > 0) {
        const avg = chunkLatencies.reduce((a, b) => a + b, 0) / chunkLatencies.length;
        const audioPos = (i + 1) * CHUNK_SIZE_MS / 1000;
        console.log(`    Chunk ${i + 1} (audio @ ${audioPos.toFixed(1)}s): avg latency ${avg.toFixed(0)}ms`);
      }
    }
  }
  
  // ============== WORD-LEVEL ANALYSIS ==============
  if (allWords.length > 0) {
    console.log('\n📝 WORD-LEVEL LATENCY');
    console.log('─'.repeat(50));
    
    const wordLatencies = allWords.map(w => w.latencyMs);
    printDetailedStats('Word latency', wordLatencies, 'ms', 1);
    
    // Word confidence
    const wordProbs = allWords.map(w => w.probability).filter(p => p > 0);
    if (wordProbs.length > 0) {
      console.log('\n  Word confidence:');
      const probStats = calculateStats(wordProbs);
      console.log(`    Mean: ${(probStats.mean * 100).toFixed(1)}%, Min: ${(probStats.min * 100).toFixed(1)}%, P5: ${(probStats.p5 * 100).toFixed(1)}%`);
    }
    
    // Sample words
    console.log('\n  Sample words (first 10):');
    for (const w of allWords.slice(0, 10)) {
      console.log(`    "${w.word}" @ ${w.audioStartS.toFixed(2)}s, latency: ${w.latencyMs.toFixed(0)}ms, conf: ${(w.probability * 100).toFixed(0)}%`);
    }
  }
  
  // ============== OUTLIER ANALYSIS ==============
  console.log('\n🔍 OUTLIER ANALYSIS (IQR method)');
  console.log('─'.repeat(50));
  
  const rtfOutliers = findOutliers(overallRTFs);
  console.log(`\n  Overall RTF outliers: ${rtfOutliers.count}/${results.length}`);
  console.log(`    IQR: ${rtfOutliers.iqr.toFixed(3)}, Bounds: [${rtfOutliers.lowerBound.toFixed(3)}, ${rtfOutliers.upperBound.toFixed(3)}]`);
  if (rtfOutliers.outliers.length > 0) {
    console.log(`    Worst: ${rtfOutliers.outliers.slice(0, 5).map(o => o.toFixed(3)).join(', ')}`);
  }
  
  const latOutliers = findOutliers(latencies);
  console.log(`\n  Latency outliers: ${latOutliers.count}/${latencies.length}`);
  console.log(`    IQR: ${latOutliers.iqr.toFixed(1)}ms, Bounds: [${latOutliers.lowerBound.toFixed(1)}, ${latOutliers.upperBound.toFixed(1)}]ms`);
  if (latOutliers.outliers.length > 0) {
    console.log(`    Worst: ${latOutliers.outliers.slice(0, 5).map(o => o.toFixed(0) + 'ms').join(', ')}`);
  }
  
  // ============== DISTRIBUTION ==============
  console.log('\n📊 RTF DISTRIBUTION');
  console.log('─'.repeat(50));
  printHistogram(overallRTFs, 8, 'x');
  
  console.log('\n📊 LATENCY DISTRIBUTION');
  console.log('─'.repeat(50));
  printHistogram(latencies, 8, 'ms');
  
  // ============== WORST STREAMS ==============
  console.log('\n📋 WORST 10 STREAMS (by RTF)');
  console.log('─'.repeat(70));
  console.log('  Stream |  RTF   | Total Time | Chunks | Words | Errors');
  console.log('─'.repeat(70));
  
  const sortedResults = [...results].sort((a, b) => b.overallRTF - a.overallRTF);
  for (const r of sortedResults.slice(0, 10)) {
    console.log(`  ${String(r.streamId).padStart(6)} | ${r.overallRTF.toFixed(3).padStart(6)} | ${(r.totalTimeMs/1000).toFixed(2).padStart(9)}s | ${String(r.chunks.length).padStart(6)} | ${String(r.words.length).padStart(5)} | ${r.errors.length}`);
  }
  
  // ============== ERRORS ==============
  if (allErrors.length > 0) {
    console.log('\n❌ ERRORS');
    console.log('─'.repeat(50));
    const errorCounts = {};
    for (const e of allErrors) {
      errorCounts[e.error] = (errorCounts[e.error] || 0) + 1;
    }
    for (const [err, count] of Object.entries(errorCounts).sort((a, b) => b[1] - a[1]).slice(0, 5)) {
      console.log(`  ${count}x: ${err}`);
    }
  }
  
  // ============== FINAL VERDICT ==============
  console.log('\n' + '═'.repeat(70));
  console.log('  FINAL VERDICT');
  console.log('═'.repeat(70));
  
  const rtfStats = calculateStats(overallRTFs);
  const latStats = calculateStats(latencies);
  
  const checks = [];
  
  // Check 1: Mean RTF < 1.0
  if (rtfStats.mean < 1.0) {
    checks.push({ pass: true, msg: `Mean RTF ${rtfStats.mean.toFixed(3)} < 1.0 (faster than real-time)` });
  } else {
    checks.push({ pass: false, msg: `Mean RTF ${rtfStats.mean.toFixed(3)} >= 1.0 (SLOWER than real-time)` });
  }
  
  // Check 2: P95 RTF < 1.5
  if (rtfStats.p95 < 1.5) {
    checks.push({ pass: true, msg: `P95 RTF ${rtfStats.p95.toFixed(3)} < 1.5` });
  } else {
    checks.push({ pass: false, msg: `P95 RTF ${rtfStats.p95.toFixed(3)} >= 1.5 (high tail latency)` });
  }
  
  // Check 3: >90% streams faster than real-time
  const pctFaster = fasterCount / results.length * 100;
  if (pctFaster >= 90) {
    checks.push({ pass: true, msg: `${pctFaster.toFixed(1)}% streams faster than real-time` });
  } else {
    checks.push({ pass: false, msg: `Only ${pctFaster.toFixed(1)}% streams faster than real-time (need 90%)` });
  }
  
  // Check 4: No errors
  if (allErrors.length === 0) {
    checks.push({ pass: true, msg: `No errors` });
  } else {
    checks.push({ pass: false, msg: `${allErrors.length} errors occurred` });
  }
  
  // Check 5: P99 latency reasonable
  if (latStats.p99 < STREAM_DURATION_S * 1000) {
    checks.push({ pass: true, msg: `P99 latency ${latStats.p99.toFixed(0)}ms is reasonable` });
  } else {
    checks.push({ pass: false, msg: `P99 latency ${latStats.p99.toFixed(0)}ms is very high` });
  }
  
  const allPassed = checks.every(c => c.pass);
  
  console.log('\n  Checks:');
  for (const c of checks) {
    console.log(`    ${c.pass ? '✅' : '❌'} ${c.msg}`);
  }
  
  console.log('\n  ' + (allPassed ? '🎉 ALL CHECKS PASSED - System handles 100 concurrent streams well!' 
                                  : '⚠️  SOME CHECKS FAILED - See issues above'));
  
  console.log('\n' + '═'.repeat(70));
  
  // Exit code
  if (!allPassed && rtfStats.mean >= 1.0) {
    process.exit(1);
  }
}

// ============== STATISTICS HELPERS ==============

function calculateStats(arr) {
  if (arr.length === 0) {
    return { min: 0, max: 0, mean: 0, median: 0, stdDev: 0, p5: 0, p25: 0, p75: 0, p95: 0, p99: 0 };
  }
  
  const sorted = [...arr].sort((a, b) => a - b);
  const sum = arr.reduce((a, b) => a + b, 0);
  const mean = sum / arr.length;
  
  const squaredDiffs = arr.map(x => Math.pow(x - mean, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / arr.length;
  const stdDev = Math.sqrt(variance);
  
  const percentile = (p) => sorted[Math.floor(sorted.length * p / 100)] || sorted[sorted.length - 1];
  
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean,
    median: percentile(50),
    stdDev,
    p5: percentile(5),
    p25: percentile(25),
    p75: percentile(75),
    p95: percentile(95),
    p99: percentile(99),
  };
}

function printDetailedStats(name, arr, unit, decimals) {
  const stats = calculateStats(arr);
  console.log(`  ${name}:`);
  console.log(`    Min:     ${stats.min.toFixed(decimals)} ${unit}`);
  console.log(`    P5:      ${stats.p5.toFixed(decimals)} ${unit}`);
  console.log(`    P25:     ${stats.p25.toFixed(decimals)} ${unit}`);
  console.log(`    Median:  ${stats.median.toFixed(decimals)} ${unit}`);
  console.log(`    Mean:    ${stats.mean.toFixed(decimals)} ${unit}`);
  console.log(`    P75:     ${stats.p75.toFixed(decimals)} ${unit}`);
  console.log(`    P95:     ${stats.p95.toFixed(decimals)} ${unit}`);
  console.log(`    P99:     ${stats.p99.toFixed(decimals)} ${unit}`);
  console.log(`    Max:     ${stats.max.toFixed(decimals)} ${unit}`);
  console.log(`    StdDev:  ${stats.stdDev.toFixed(decimals)} ${unit}`);
}

function findOutliers(arr) {
  if (arr.length < 4) {
    return { count: 0, outliers: [], iqr: 0, lowerBound: 0, upperBound: 0 };
  }
  
  const sorted = [...arr].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = q3 - q1;
  
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;
  
  const outliers = arr.filter(x => x < lowerBound || x > upperBound).sort((a, b) => b - a);
  
  return { count: outliers.length, outliers, iqr, lowerBound, upperBound, q1, q3 };
}

function printHistogram(arr, bins, unit) {
  if (arr.length === 0) return;
  
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = max - min || 1;
  const binSize = range / bins;
  
  const buckets = Array(bins).fill(0);
  for (const val of arr) {
    const idx = Math.min(Math.floor((val - min) / binSize), bins - 1);
    buckets[idx]++;
  }
  
  const maxCount = Math.max(...buckets);
  const barWidth = 35;
  
  for (let i = 0; i < bins; i++) {
    const binStart = min + i * binSize;
    const binEnd = min + (i + 1) * binSize;
    const count = buckets[i];
    const bar = '█'.repeat(Math.round(count / maxCount * barWidth));
    const pct = (count / arr.length * 100).toFixed(1);
    console.log(`  ${binStart.toFixed(2).padStart(7)}-${binEnd.toFixed(2).padEnd(7)} ${unit} |${bar.padEnd(barWidth)}| ${String(count).padStart(4)} (${pct.padStart(5)}%)`);
  }
}
