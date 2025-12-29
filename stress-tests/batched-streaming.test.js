#!/usr/bin/env node
/**
 * Batched Streaming Stress Test
 * 
 * Tests the StreamingBatcher for high-throughput concurrent transcription.
 * Uses a single Engine shared across all streams with batched processing.
 * 
 * Run with: node stress-tests/batched-streaming.test.js [options]
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
const BATCH_SIZE = parseInt(args.batch || '16');
const MAX_WAIT_MS = parseInt(args.wait || '25');
const MODEL_PATH = args.model || './models/tiny';
const VERBOSE = args.verbose === true;

const SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = Math.floor(SAMPLE_RATE * CHUNK_SIZE_MS / 1000);

async function main() {
  console.log('═'.repeat(70));
  console.log('  BATCHED STREAMING STRESS TEST');
  console.log('═'.repeat(70));
  console.log(`  Concurrent streams:  ${CONCURRENT_STREAMS}`);
  console.log(`  Stream duration:     ${STREAM_DURATION_S}s`);
  console.log(`  Chunk size:          ${CHUNK_SIZE_MS}ms`);
  console.log(`  Batch size:          ${BATCH_SIZE}`);
  console.log(`  Max wait:            ${MAX_WAIT_MS}ms`);
  console.log(`  Model:               ${MODEL_PATH}`);
  console.log('═'.repeat(70));
  
  // Verify prerequisites
  if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
    console.error('❌ Model not found:', MODEL_PATH);
    process.exit(1);
  }
  
  const { StreamingBatcher, createBatcher } = require('../streaming-batcher.js');
  const { decodeAudio } = require('../index.js');
  
  // Load audio files
  const audioFiles = [
    './tests/fixtures/hello.wav',
    './tests/fixtures/numbers.wav',
    './tests/fixtures/sentence.wav',
  ].filter(f => fs.existsSync(f));
  
  console.log(`\n📁 Loading ${audioFiles.length} audio files...`);
  
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
  
  console.log(`  Total audio: ${(streamAudio.length / SAMPLE_RATE).toFixed(2)}s`);
  
  // Create the batcher
  console.log(`\n🔧 Creating StreamingBatcher...`);
  const batcher = createBatcher(MODEL_PATH, {
    maxBatchSize: BATCH_SIZE,
    maxWaitMs: MAX_WAIT_MS,
    language: 'en',
    beamSize: 5,
    wordTimestamps: false,
  });
  
  // Track batch processing
  let batchCount = 0;
  batcher.on('batchComplete', ({ size, durationMs }) => {
    batchCount++;
    if (VERBOSE) {
      console.log(`  Batch ${batchCount}: ${size} chunks in ${durationMs}ms (${(durationMs/size).toFixed(0)}ms/chunk)`);
    }
  });
  
  console.log(`\n🚀 Starting ${CONCURRENT_STREAMS} streams...`);
  const testStartTime = Date.now();
  
  // Create all streams and their chunk processing promises
  const streamResults = [];
  const streamPromises = [];
  
  for (let i = 0; i < CONCURRENT_STREAMS; i++) {
    const streamId = batcher.createStream();
    const streamResult = {
      streamId,
      chunks: [],
      startTime: Date.now(),
      endTime: null,
    };
    streamResults.push(streamResult);
    
    // Process all chunks for this stream
    const chunkPromises = [];
    const totalChunks = Math.ceil(streamAudio.length / CHUNK_SAMPLES);
    
    for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
      const chunkStart = chunkIdx * CHUNK_SAMPLES;
      const chunkEnd = Math.min((chunkIdx + 1) * CHUNK_SAMPLES, streamAudio.length);
      const chunkAudio = streamAudio.slice(chunkStart, chunkEnd);
      const audioStartS = chunkStart / SAMPLE_RATE;
      
      const chunkPromise = (async () => {
        const submitTime = Date.now();
        try {
          const result = await batcher.transcribeChunk(streamId, chunkAudio, audioStartS);
          const wallClockSinceStart = Date.now() - streamResult.startTime;
          const audioEndS = audioStartS + result.audioDurationS;
          const latencyVsRealtimeMs = wallClockSinceStart - (audioEndS * 1000);
          const realTimeFactor = result.processingTimeMs / (result.audioDurationS * 1000);
          
          streamResult.chunks.push({
            chunkIdx,
            audioStartS,
            audioEndS,
            audioDurationMs: result.audioDurationS * 1000,
            processingTimeMs: result.processingTimeMs,
            queueTimeMs: result.queueTimeMs,
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
      })();
      
      chunkPromises.push(chunkPromise);
    }
    
    // Wait for all chunks in this stream
    const streamPromise = Promise.all(chunkPromises).then(() => {
      streamResult.endTime = Date.now();
      batcher.closeStream(streamId);
    });
    
    streamPromises.push(streamPromise);
  }
  
  // Progress reporting
  const progressInterval = setInterval(() => {
    const completed = streamResults.filter(r => r.endTime !== null).length;
    const pct = (completed / CONCURRENT_STREAMS * 100).toFixed(0);
    const stats = batcher.getStats();
    process.stdout.write(`\r  Progress: ${completed}/${CONCURRENT_STREAMS} (${pct}%) | Batches: ${stats.totalBatches} | Queue: ${stats.pendingRequests}`);
  }, 100);
  
  // Wait for all streams to complete
  await Promise.all(streamPromises);
  
  clearInterval(progressInterval);
  const totalTestTime = Date.now() - testStartTime;
  
  console.log('\n\n');
  
  // Get final stats
  const finalStats = batcher.getStats();
  
  // Cleanup
  batcher.destroy();
  
  // Analyze results
  analyzeResults(streamResults, totalTestTime, finalStats);
}

function analyzeResults(results, totalTestTime, batcherStats) {
  console.log('═'.repeat(70));
  console.log('  COMPREHENSIVE STATISTICAL ANALYSIS');
  console.log('═'.repeat(70));
  
  const allChunks = results.flatMap(r => r.chunks.filter(c => !c.error));
  const allErrors = results.flatMap(r => r.chunks.filter(c => c.error));
  
  // ============== OVERVIEW ==============
  console.log('\n📊 OVERVIEW');
  console.log('─'.repeat(50));
  console.log(`  Total streams:           ${results.length}`);
  console.log(`  Total test time:         ${(totalTestTime / 1000).toFixed(2)}s`);
  console.log(`  Total chunks:            ${allChunks.length}`);
  console.log(`  Total batches:           ${batcherStats.totalBatches}`);
  console.log(`  Avg batch size:          ${batcherStats.avgBatchSize.toFixed(1)}`);
  console.log(`  Total errors:            ${allErrors.length}`);
  console.log(`  Queue high water mark:   ${batcherStats.queueHighWaterMark}`);
  
  // ============== OVERALL REAL-TIME FACTOR ==============
  console.log('\n⏱️  OVERALL REAL-TIME FACTOR (per stream)');
  console.log('─'.repeat(50));
  console.log('  RTF < 1.0 means faster than real-time (good)');
  console.log('  RTF > 1.0 means slower than real-time (bad)');
  
  const overallRTFs = results.map(r => {
    const totalTimeMs = r.endTime - r.startTime;
    const audioDurationS = r.chunks.reduce((sum, c) => sum + (c.audioDurationMs || 0), 0) / 1000;
    return totalTimeMs / (audioDurationS * 1000);
  });
  
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
  
  // ============== QUEUE TIME ==============
  console.log('\n⏳ QUEUE TIME (time waiting before processing)');
  console.log('─'.repeat(50));
  
  const queueTimes = allChunks.map(c => c.queueTimeMs);
  printDetailedStats('Queue time', queueTimes, 'ms', 1);
  
  // ============== END-TO-END LATENCY ==============
  console.log('\n📈 END-TO-END LATENCY (vs real-time audio position)');
  console.log('─'.repeat(50));
  console.log('  Latency = wall_clock_time - audio_position');
  console.log('  Negative = ahead of real-time, Positive = behind');
  
  const latencies = allChunks.map(c => c.latencyVsRealtimeMs);
  printDetailedStats('Latency', latencies, 'ms', 1);
  
  // ============== THROUGHPUT ==============
  console.log('\n📈 THROUGHPUT');
  console.log('─'.repeat(50));
  
  const totalAudioS = allChunks.reduce((sum, c) => sum + (c.audioDurationMs || 0), 0) / 1000;
  const throughputRatio = totalAudioS / (totalTestTime / 1000);
  
  console.log(`  Total audio processed:   ${totalAudioS.toFixed(1)}s`);
  console.log(`  Total wall clock time:   ${(totalTestTime/1000).toFixed(2)}s`);
  console.log(`  Throughput ratio:        ${throughputRatio.toFixed(2)}x real-time`);
  console.log(`  Chunks per second:       ${(allChunks.length / (totalTestTime/1000)).toFixed(1)}`);
  
  // ============== DISTRIBUTION ==============
  console.log('\n📊 RTF DISTRIBUTION');
  console.log('─'.repeat(50));
  printHistogram(overallRTFs, 8, 'x');
  
  console.log('\n📊 LATENCY DISTRIBUTION');
  console.log('─'.repeat(50));
  printHistogram(latencies, 8, 'ms');
  
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
  
  // Check 4: Throughput > 1x
  if (throughputRatio > 1.0) {
    checks.push({ pass: true, msg: `Throughput ${throughputRatio.toFixed(2)}x > 1.0 real-time` });
  } else {
    checks.push({ pass: false, msg: `Throughput ${throughputRatio.toFixed(2)}x < 1.0 real-time` });
  }
  
  // Check 5: No errors
  if (allErrors.length === 0) {
    checks.push({ pass: true, msg: `No errors` });
  } else {
    checks.push({ pass: false, msg: `${allErrors.length} errors occurred` });
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
    console.log(`  ${binStart.toFixed(2).padStart(7)}-${binEnd.toFixed(2).padEnd(7)} ${unit} |${bar.padEnd(barWidth)}| ${String(count).padStart(4)} (${pct.padStart(5)}%)`);
  }
}

main().catch(console.error);
