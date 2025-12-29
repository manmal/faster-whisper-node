#!/usr/bin/env node
/**
 * Memory Stability Stress Test
 * 
 * Runs many transcriptions and monitors memory usage to detect leaks.
 * 
 * Run with: node stress-tests/memory.test.js [options]
 * 
 * Options:
 *   --iterations=N   Number of iterations (default: 100)
 *   --gc             Force garbage collection between batches (requires --expose-gc)
 *   --verbose        Show detailed progress
 */

const path = require('path');
const fs = require('fs');

// Parse command line args
const args = {};
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=');
  args[key] = value || true;
}

const ITERATIONS = parseInt(args.iterations || '100');
const FORCE_GC = args.gc === true;
const VERBOSE = args.verbose === true;
const MODEL_PATH = args.model || './models/tiny';
const AUDIO_PATH = args.audio || './tests/fixtures/hello.wav';

// Verify files exist
if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
  console.error('❌ Model not found:', MODEL_PATH);
  process.exit(1);
}
if (!fs.existsSync(AUDIO_PATH)) {
  console.error('❌ Audio file not found:', AUDIO_PATH);
  process.exit(1);
}

// Load audio buffer for buffer-based tests
const audioBuffer = fs.readFileSync(AUDIO_PATH);

const { Engine, decodeAudio } = require('../index');

function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(2)} ${units[unit]}`;
}

function getMemoryStats() {
  const mem = process.memoryUsage();
  return {
    heapUsed: mem.heapUsed,
    heapTotal: mem.heapTotal,
    external: mem.external,
    rss: mem.rss,
  };
}

async function runTest() {
  console.log('='.repeat(60));
  console.log('Memory Stability Stress Test');
  console.log('='.repeat(60));
  console.log(`Iterations:      ${ITERATIONS}`);
  console.log(`Model:           ${MODEL_PATH}`);
  console.log(`Audio:           ${AUDIO_PATH}`);
  console.log(`Force GC:        ${FORCE_GC ? 'yes (every 10 iterations)' : 'no'}`);
  
  if (FORCE_GC && !global.gc) {
    console.warn('⚠️  --gc requires: node --expose-gc stress-tests/memory.test.js --gc');
  }
  
  console.log('\nLoading engine...');
  const engine = new Engine(MODEL_PATH);
  
  // Initial memory baseline
  if (FORCE_GC && global.gc) global.gc();
  const initialMem = getMemoryStats();
  console.log(`Initial memory: ${formatBytes(initialMem.heapUsed)} heap, ${formatBytes(initialMem.rss)} RSS`);
  
  const memorySnapshots = [{ iteration: 0, ...initialMem }];
  const transcriptionTimes = [];
  
  console.log(`\nRunning ${ITERATIONS} transcriptions...\n`);
  
  const audioFiles = [
    './tests/fixtures/hello.wav',
    './tests/fixtures/numbers.wav',
    './tests/fixtures/sentence.wav',
  ].filter(f => fs.existsSync(f));
  
  const startTime = Date.now();
  
  for (let i = 0; i < ITERATIONS; i++) {
    const iterStart = Date.now();
    
    // Alternate between different transcription methods
    const method = i % 3;
    const audioFile = audioFiles[i % audioFiles.length];
    
    try {
      switch (method) {
        case 0:
          // transcribeFile
          engine.transcribeFile(audioFile);
          break;
        case 1:
          // transcribeBuffer
          engine.transcribeBuffer(audioBuffer);
          break;
        case 2:
          // transcribe (legacy)
          engine.transcribe(audioFile);
          break;
      }
    } catch (err) {
      console.error(`\n❌ Iteration ${i + 1} failed:`, err.message);
    }
    
    transcriptionTimes.push(Date.now() - iterStart);
    
    // Progress
    if ((i + 1) % 10 === 0 || i === ITERATIONS - 1) {
      // Force GC if requested
      if (FORCE_GC && global.gc) {
        global.gc();
      }
      
      const mem = getMemoryStats();
      memorySnapshots.push({ iteration: i + 1, ...mem });
      
      const heapGrowth = mem.heapUsed - initialMem.heapUsed;
      const rssGrowth = mem.rss - initialMem.rss;
      
      if (VERBOSE) {
        console.log(`Iteration ${i + 1}: heap=${formatBytes(mem.heapUsed)} (+${formatBytes(heapGrowth)}), RSS=${formatBytes(mem.rss)} (+${formatBytes(rssGrowth)})`);
      } else {
        process.stdout.write(`\rProgress: ${i + 1}/${ITERATIONS} - heap: ${formatBytes(mem.heapUsed)} (+${formatBytes(heapGrowth)})`);
      }
    }
  }
  
  const elapsed = Date.now() - startTime;
  console.log('\n');
  
  // Final memory check
  if (FORCE_GC && global.gc) {
    global.gc();
    await new Promise(r => setTimeout(r, 100)); // Let GC complete
    global.gc();
  }
  
  const finalMem = getMemoryStats();
  const heapGrowth = finalMem.heapUsed - initialMem.heapUsed;
  const rssGrowth = finalMem.rss - initialMem.rss;
  
  // Calculate timing stats
  const sortedTimes = [...transcriptionTimes].sort((a, b) => a - b);
  const avgTime = transcriptionTimes.reduce((a, b) => a + b, 0) / transcriptionTimes.length;
  const medianTime = sortedTimes[Math.floor(sortedTimes.length / 2)];
  
  console.log('='.repeat(60));
  console.log('Results');
  console.log('='.repeat(60));
  console.log(`Iterations:          ${ITERATIONS}`);
  console.log(`Total time:          ${elapsed} ms`);
  console.log(`Throughput:          ${(ITERATIONS / elapsed * 1000).toFixed(2)} transcriptions/s`);
  
  console.log(`\nTranscription timing:`);
  console.log(`  Min:               ${Math.min(...transcriptionTimes)} ms`);
  console.log(`  Max:               ${Math.max(...transcriptionTimes)} ms`);
  console.log(`  Mean:              ${avgTime.toFixed(0)} ms`);
  console.log(`  Median:            ${medianTime} ms`);
  
  console.log(`\nMemory usage:`);
  console.log(`  Initial heap:      ${formatBytes(initialMem.heapUsed)}`);
  console.log(`  Final heap:        ${formatBytes(finalMem.heapUsed)}`);
  console.log(`  Heap growth:       ${formatBytes(heapGrowth)} (${(heapGrowth / initialMem.heapUsed * 100).toFixed(1)}%)`);
  console.log(`  Initial RSS:       ${formatBytes(initialMem.rss)}`);
  console.log(`  Final RSS:         ${formatBytes(finalMem.rss)}`);
  console.log(`  RSS growth:        ${formatBytes(rssGrowth)}`);
  
  // Analyze memory trend
  const heapHistory = memorySnapshots.map(s => s.heapUsed);
  const heapTrend = heapHistory.length > 1 
    ? (heapHistory[heapHistory.length - 1] - heapHistory[0]) / heapHistory.length
    : 0;
  
  console.log(`\nMemory trend analysis:`);
  console.log(`  Avg heap growth/10 iters: ${formatBytes(heapTrend * 10)}`);
  
  // Check for potential leaks
  const leakThreshold = 50 * 1024 * 1024; // 50MB
  const growthPerIteration = heapGrowth / ITERATIONS;
  
  if (heapGrowth > leakThreshold) {
    console.log(`\n⚠️  Warning: Significant heap growth detected (${formatBytes(heapGrowth)})`);
    console.log(`   This may indicate a memory leak.`);
    console.log(`   Growth per iteration: ${formatBytes(growthPerIteration)}`);
  } else {
    console.log(`\n✅ Memory usage appears stable`);
  }
  
  // Memory timeline
  console.log(`\nMemory timeline (heap used):`);
  const maxBarWidth = 40;
  const maxHeap = Math.max(...heapHistory);
  for (const snapshot of memorySnapshots) {
    const barWidth = Math.round((snapshot.heapUsed / maxHeap) * maxBarWidth);
    const bar = '█'.repeat(barWidth) + '░'.repeat(maxBarWidth - barWidth);
    console.log(`  ${String(snapshot.iteration).padStart(4)}: ${bar} ${formatBytes(snapshot.heapUsed)}`);
  }
  
  console.log('\n✅ Memory test complete!');
}

runTest().catch(console.error);
