#!/usr/bin/env node
/**
 * Realtime Streaming Stress Test
 * 
 * Simulates REAL streaming where audio chunks arrive in real-time.
 * Each stream submits chunks at the actual audio rate (e.g., 2s chunk every 2s).
 * This tests if we can keep up with 100 concurrent real-time streams.
 * 
 * Run with: node stress-tests/realtime-streaming.test.js [options]
 */

const path = require('path');
const fs = require('fs');

// Parse command line args
const args = {};
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=');
  args[key] = value || true;
}

const CONCURRENT_STREAMS = parseInt(args.streams || '100');
const STREAM_DURATION_S = parseFloat(args.duration || '10');
const CHUNK_SIZE_MS = parseInt(args.chunk || '2000');
const NUM_WORKERS = parseInt(args.workers || '8');
const MODEL_PATH = args.model || './models/tiny';
const VERBOSE = args.verbose === true;

const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SIZE_MS / 1000);

async function main() {
  console.log('═'.repeat(70));
  console.log('  REALTIME STREAMING STRESS TEST');
  console.log('═'.repeat(70));
  console.log(`  Concurrent streams:  ${CONCURRENT_STREAMS}`);
  console.log(`  Stream duration:     ${STREAM_DURATION_S}s`);
  console.log(`  Chunk size:          ${CHUNK_SIZE_MS}ms`);
  console.log(`  Worker threads:      ${NUM_WORKERS}`);
  console.log(`  Model:               ${MODEL_PATH}`);
  console.log('═'.repeat(70));
  console.log('');
  console.log('  📢 This test simulates REAL streaming where chunks arrive at');
  console.log('     the actual audio rate. A stream sending 2s chunks will');
  console.log('     submit a new chunk every ~2 seconds.');
  console.log('');
  
  // Verify prerequisites
  if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
    console.error('❌ Model not found:', MODEL_PATH);
    process.exit(1);
  }
  
  const { WorkerPoolBatcher, createWorkerPool } = require('../worker-pool-batcher.js');
  const { decodeAudio } = require('../index.js');
  
  // Load audio files
  const audioFiles = [
    './tests/fixtures/hello.wav',
    './tests/fixtures/numbers.wav',
    './tests/fixtures/sentence.wav',
  ].filter(f => fs.existsSync(f));
  
  console.log(`📁 Loading ${audioFiles.length} audio files...`);
  
  const allSamples = [];
  for (const f of audioFiles) {
    const samples = decodeAudio(f);
    allSamples.push(...samples);
  }
  
  // Extend audio to target duration
  const targetSamples = Math.ceil(STREAM_DURATION_S * SAMPLE_RATE);
  const extendedSamples = [];
  while (extendedSamples.length < targetSamples) {
    extendedSamples.push(...allSamples);
  }
  const streamAudio = new Float32Array(extendedSamples.slice(0, targetSamples));
  
  console.log(`  Total audio: ${(streamAudio.length / SAMPLE_RATE).toFixed(2)}s per stream`);
  
  // Create the worker pool
  console.log(`\n🔧 Creating WorkerPoolBatcher with ${NUM_WORKERS} workers...`);
  const pool = createWorkerPool(MODEL_PATH, {
    numWorkers: NUM_WORKERS,
    language: 'en',
    beamSize: 5,
    wordTimestamps: false,
  });
  
  // Wait for workers to initialize
  console.log(`  Initializing workers...`);
  await pool.init();
  console.log(`  All ${NUM_WORKERS} workers ready`);
  
  console.log(`\n🚀 Starting ${CONCURRENT_STREAMS} realtime streams...`);
  console.log(`   Each stream submits chunks every ${CHUNK_SIZE_MS}ms (real-time rate)`);
  
  const testStartTime = Date.now();
  
  // Create all streams
  const streamResults = [];
  const streamPromises = [];
  const totalChunks = Math.ceil(streamAudio.length / CHUNK_SAMPLES);
  
  for (let i = 0; i < CONCURRENT_STREAMS; i++) {
    const streamId = pool.createStream();
    const streamResult = {
      streamId,
      chunks: [],
      startTime: Date.now(),
      endTime: null,
    };
    streamResults.push(streamResult);
    
    // Create async generator that yields chunks at real-time rate
    const streamPromise = (async () => {
      for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
        const chunkStart = chunkIdx * CHUNK_SAMPLES;
        const chunkEnd = Math.min((chunkIdx + 1) * CHUNK_SAMPLES, streamAudio.length);
        const chunkAudio = streamAudio.slice(chunkStart, chunkEnd);
        const audioStartS = chunkStart / SAMPLE_RATE;
        const audioEndS = chunkEnd / SAMPLE_RATE;
        
        // Wait until it's time to submit this chunk (simulating real-time audio arrival)
        const expectedArrivalTime = streamResult.startTime + (audioStartS * 1000);
        const now = Date.now();
        if (expectedArrivalTime > now) {
          await new Promise(r => setTimeout(r, expectedArrivalTime - now));
        }
        
        const submitTime = Date.now();
        
        try {
          const result = await pool.transcribeChunk(streamId, chunkAudio, audioStartS);
          const completionTime = Date.now();
          const wallClockSinceStart = completionTime - streamResult.startTime;
          const latencyVsRealtimeMs = wallClockSinceStart - (audioEndS * 1000);
          const realTimeFactor = result.processingTimeMs / (result.audioDurationS * 1000);
          
          streamResult.chunks.push({
            chunkIdx,
            audioStartS,
            audioEndS,
            audioDurationMs: result.audioDurationS * 1000,
            processingTimeMs: result.processingTimeMs,
            queueTimeMs: result.queueTimeMs,
            submitToCompleteMs: completionTime - submitTime,
            wallClockSinceStartMs: wallClockSinceStart,
            latencyVsRealtimeMs,
            realTimeFactor,
            text: result.text.trim().substring(0, 50),
          });
        } catch (error) {
          streamResult.chunks.push({
            chunkIdx,
            error: error.message,
          });
        }
      }
      
      streamResult.endTime = Date.now();
      pool.closeStream(streamId);
    })();
    
    streamPromises.push(streamPromise);
  }
  
  // Progress reporting
  const progressInterval = setInterval(() => {
    const completed = streamResults.filter(r => r.endTime !== null).length;
    const totalCompleted = streamResults.reduce((sum, r) => sum + r.chunks.length, 0);
    const pct = (completed / CONCURRENT_STREAMS * 100).toFixed(0);
    const expectedChunks = CONCURRENT_STREAMS * totalChunks;
    const stats = pool.getStats();
    const elapsed = ((Date.now() - testStartTime) / 1000).toFixed(1);
    process.stdout.write(`\r  [${elapsed}s] Streams: ${completed}/${CONCURRENT_STREAMS} (${pct}%) | Chunks: ${totalCompleted}/${expectedChunks} | Pending: ${stats.pendingRequests}`);
  }, 200);
  
  // Wait for all streams to complete
  await Promise.all(streamPromises);
  
  clearInterval(progressInterval);
  const totalTestTime = Date.now() - testStartTime;
  
  console.log('\n\n');
  
  // Get final stats
  const finalStats = pool.getStats();
  
  // Cleanup
  pool.destroy();
  
  // Analyze results
  analyzeResults(streamResults, totalTestTime, finalStats, STREAM_DURATION_S);
}

function analyzeResults(results, totalTestTime, poolStats, expectedDuration) {
  console.log('═'.repeat(70));
  console.log('  COMPREHENSIVE STATISTICAL ANALYSIS');
  console.log('═'.repeat(70));
  
  const allChunks = results.flatMap(r => r.chunks.filter(c => !c.error));
  const allErrors = results.flatMap(r => r.chunks.filter(c => c.error));
  
  // ============== OVERVIEW ==============
  console.log('\n📊 OVERVIEW');
  console.log('─'.repeat(50));
  console.log(`  Total streams:           ${results.length}`);
  console.log(`  Expected duration:       ${expectedDuration.toFixed(1)}s per stream`);
  console.log(`  Actual test time:        ${(totalTestTime / 1000).toFixed(2)}s`);
  console.log(`  Total chunks:            ${allChunks.length}`);
  console.log(`  Workers used:            ${poolStats.workersReady}`);
  console.log(`  Total errors:            ${allErrors.length}`);
  
  // ============== LATENCY (KEY METRIC for streaming) ==============
  console.log('\n📈 STREAMING LATENCY (KEY METRIC)');
  console.log('─'.repeat(50));
  console.log('  Latency = time_result_ready - time_audio_ended');
  console.log('  < 0ms: result ready before audio ended (impossible in real-time)');
  console.log('  0-500ms: excellent');
  console.log('  500-1000ms: good');
  console.log('  1000-2000ms: acceptable');
  console.log('  > 2000ms: poor (falling behind)');
  
  const latencies = allChunks.map(c => c.latencyVsRealtimeMs);
  printDetailedStats('Latency', latencies, 'ms', 0);
  
  // Latency buckets
  const buckets = {
    excellent: latencies.filter(l => l >= 0 && l < 500).length,
    good: latencies.filter(l => l >= 500 && l < 1000).length,
    acceptable: latencies.filter(l => l >= 1000 && l < 2000).length,
    poor: latencies.filter(l => l >= 2000).length,
    ahead: latencies.filter(l => l < 0).length,
  };
  
  console.log('\n  Latency distribution:');
  console.log(`    ⚡ < 0ms (ahead):     ${buckets.ahead} (${(buckets.ahead/latencies.length*100).toFixed(1)}%)`);
  console.log(`    ✅ 0-500ms:           ${buckets.excellent} (${(buckets.excellent/latencies.length*100).toFixed(1)}%)`);
  console.log(`    👍 500-1000ms:        ${buckets.good} (${(buckets.good/latencies.length*100).toFixed(1)}%)`);
  console.log(`    ⚠️  1000-2000ms:       ${buckets.acceptable} (${(buckets.acceptable/latencies.length*100).toFixed(1)}%)`);
  console.log(`    ❌ > 2000ms:          ${buckets.poor} (${(buckets.poor/latencies.length*100).toFixed(1)}%)`);
  
  // ============== STREAM COMPLETION ==============
  console.log('\n⏱️  STREAM COMPLETION TIME');
  console.log('─'.repeat(50));
  console.log(`  Expected duration: ${expectedDuration}s per stream`);
  
  const streamDurations = results.map(r => (r.endTime - r.startTime) / 1000);
  printDetailedStats('Completion time', streamDurations, 's', 2);
  
  const onTimeStreams = streamDurations.filter(d => d <= expectedDuration * 1.5).length;
  console.log(`\n  ✅ Completed within 1.5x expected: ${onTimeStreams}/${results.length} (${(onTimeStreams/results.length*100).toFixed(1)}%)`);
  
  // ============== PROCESSING TIME ==============
  console.log('\n🔧 CHUNK PROCESSING TIME');
  console.log('─'.repeat(50));
  
  const processingTimes = allChunks.map(c => c.processingTimeMs);
  printDetailedStats('Processing time', processingTimes, 'ms', 0);
  
  // ============== QUEUE TIME ==============
  console.log('\n⏳ QUEUE TIME');
  console.log('─'.repeat(50));
  
  const queueTimes = allChunks.map(c => c.queueTimeMs);
  printDetailedStats('Queue time', queueTimes, 'ms', 0);
  
  // ============== THROUGHPUT ==============
  console.log('\n📈 THROUGHPUT');
  console.log('─'.repeat(50));
  
  const totalAudioS = allChunks.reduce((sum, c) => sum + (c.audioDurationMs || 0), 0) / 1000;
  const theoreticalMaxThroughput = poolStats.workersReady; // Each worker does ~1x real-time
  const actualThroughput = totalAudioS / (totalTestTime / 1000);
  
  console.log(`  Total audio processed:   ${totalAudioS.toFixed(1)}s`);
  console.log(`  Total wall clock time:   ${(totalTestTime/1000).toFixed(2)}s`);
  console.log(`  Actual throughput:       ${actualThroughput.toFixed(2)}x real-time`);
  console.log(`  Theoretical max:         ${theoreticalMaxThroughput}x (${poolStats.workersReady} workers)`);
  console.log(`  Efficiency:              ${(actualThroughput/theoreticalMaxThroughput*100).toFixed(1)}%`);
  
  // ============== DISTRIBUTION ==============
  console.log('\n📊 LATENCY DISTRIBUTION');
  console.log('─'.repeat(50));
  printHistogram(latencies, 10, 'ms');
  
  // ============== FINAL VERDICT ==============
  console.log('\n' + '═'.repeat(70));
  console.log('  FINAL VERDICT');
  console.log('═'.repeat(70));
  
  const latStats = calculateStats(latencies);
  const durationStats = calculateStats(streamDurations);
  
  const checks = [];
  
  // Check 1: P95 latency < 2000ms
  if (latStats.p95 < 2000) {
    checks.push({ pass: true, msg: `P95 latency ${latStats.p95.toFixed(0)}ms < 2000ms (good)` });
  } else {
    checks.push({ pass: false, msg: `P95 latency ${latStats.p95.toFixed(0)}ms >= 2000ms (falling behind)` });
  }
  
  // Check 2: Mean latency < 1000ms
  if (latStats.mean < 1000) {
    checks.push({ pass: true, msg: `Mean latency ${latStats.mean.toFixed(0)}ms < 1000ms` });
  } else {
    checks.push({ pass: false, msg: `Mean latency ${latStats.mean.toFixed(0)}ms >= 1000ms` });
  }
  
  // Check 3: >80% chunks have latency < 1500ms
  const goodLatencyPct = (buckets.excellent + buckets.good + buckets.acceptable) / latencies.length * 100;
  if (goodLatencyPct >= 80) {
    checks.push({ pass: true, msg: `${goodLatencyPct.toFixed(1)}% chunks have latency < 2000ms` });
  } else {
    checks.push({ pass: false, msg: `Only ${goodLatencyPct.toFixed(1)}% chunks have latency < 2000ms (need 80%)` });
  }
  
  // Check 4: All streams complete
  const incompleteStreams = results.filter(r => r.endTime === null).length;
  if (incompleteStreams === 0) {
    checks.push({ pass: true, msg: `All ${results.length} streams completed` });
  } else {
    checks.push({ pass: false, msg: `${incompleteStreams} streams incomplete` });
  }
  
  // Check 5: No errors
  if (allErrors.length === 0) {
    checks.push({ pass: true, msg: `No errors` });
  } else {
    checks.push({ pass: false, msg: `${allErrors.length} errors occurred` });
  }
  
  // Check 6: Efficiency > 50%
  const efficiency = actualThroughput / theoreticalMaxThroughput;
  if (efficiency >= 0.5) {
    checks.push({ pass: true, msg: `Worker efficiency ${(efficiency*100).toFixed(1)}% >= 50%` });
  } else {
    checks.push({ pass: false, msg: `Worker efficiency ${(efficiency*100).toFixed(1)}% < 50% (underutilized)` });
  }
  
  const allPassed = checks.every(c => c.pass);
  
  console.log('\n  Checks:');
  for (const c of checks) {
    console.log(`    ${c.pass ? '✅' : '❌'} ${c.msg}`);
  }
  
  console.log('\n  ' + (allPassed ? '🎉 ALL CHECKS PASSED - Real-time streaming works for 100 concurrent streams!' 
                                  : '⚠️  SOME CHECKS FAILED - See issues above'));
  
  console.log('\n' + '═'.repeat(70));
  
  // Exit code
  if (!allPassed) {
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
    console.log(`  ${binStart.toFixed(0).padStart(6)}-${binEnd.toFixed(0).padEnd(6)} ${unit} |${bar.padEnd(barWidth)}| ${String(count).padStart(4)} (${pct.padStart(5)}%)`);
  }
}

main().catch(console.error);
