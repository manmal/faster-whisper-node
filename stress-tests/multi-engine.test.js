#!/usr/bin/env node
/**
 * Multi-Engine Stress Test
 * 
 * Tests running multiple Engine instances simultaneously in the same process.
 * This tests thread safety and resource sharing of the native module.
 * 
 * Run with: node stress-tests/multi-engine.test.js [options]
 * 
 * Options:
 *   --engines=N      Number of engine instances (default: 4)
 *   --requests=N     Total requests (default: 100)
 *   --sequential     Run sequentially instead of interleaved
 */

const path = require('path');
const fs = require('fs');

// Parse command line args
const args = {};
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=');
  args[key] = value || true;
}

const ENGINE_COUNT = parseInt(args.engines || '4');
const TOTAL_REQUESTS = parseInt(args.requests || '100');
const SEQUENTIAL = args.sequential === true;
const MODEL_PATH = args.model || './models/tiny';

// Check model exists
if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
  console.error('❌ Model not found:', MODEL_PATH);
  process.exit(1);
}

const AUDIO_FILES = [
  './tests/fixtures/hello.wav',
  './tests/fixtures/numbers.wav',
  './tests/fixtures/sentence.wav',
].filter(f => fs.existsSync(f));

if (AUDIO_FILES.length === 0) {
  console.error('❌ No test audio files found');
  process.exit(1);
}

const { Engine } = require('../index');

async function runTest() {
  console.log('='.repeat(60));
  console.log('Multi-Engine Stress Test');
  console.log('='.repeat(60));
  console.log(`Engine instances: ${ENGINE_COUNT}`);
  console.log(`Total requests:   ${TOTAL_REQUESTS}`);
  console.log(`Mode:             ${SEQUENTIAL ? 'sequential' : 'interleaved'}`);
  console.log(`Model:            ${MODEL_PATH}`);
  console.log('');
  
  // Create multiple engines
  console.log('Creating engines...');
  const engines = [];
  for (let i = 0; i < ENGINE_COUNT; i++) {
    const start = Date.now();
    engines.push(new Engine(MODEL_PATH));
    console.log(`  Engine ${i + 1}/${ENGINE_COUNT} created (${Date.now() - start}ms)`);
  }
  
  console.log('\nMemory after loading all engines:');
  const mem = process.memoryUsage();
  console.log(`  Heap: ${(mem.heapUsed / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  RSS:  ${(mem.rss / 1024 / 1024).toFixed(2)} MB`);
  
  const results = [];
  const engineResults = Array(ENGINE_COUNT).fill(null).map(() => ({
    success: 0,
    failed: 0,
    totalTime: 0,
  }));
  
  console.log(`\nRunning ${TOTAL_REQUESTS} requests...\n`);
  const startTime = Date.now();
  
  if (SEQUENTIAL) {
    // Each engine processes all its requests before moving to next
    const requestsPerEngine = Math.ceil(TOTAL_REQUESTS / ENGINE_COUNT);
    
    for (let e = 0; e < ENGINE_COUNT; e++) {
      const engine = engines[e];
      const count = Math.min(requestsPerEngine, TOTAL_REQUESTS - e * requestsPerEngine);
      
      for (let i = 0; i < count; i++) {
        const audioFile = AUDIO_FILES[(e * requestsPerEngine + i) % AUDIO_FILES.length];
        const reqStart = Date.now();
        
        try {
          const result = engine.transcribeFile(audioFile);
          const elapsed = Date.now() - reqStart;
          
          results.push({
            engine: e,
            success: true,
            elapsed,
            text: result.text.substring(0, 30),
          });
          
          engineResults[e].success++;
          engineResults[e].totalTime += elapsed;
        } catch (err) {
          results.push({
            engine: e,
            success: false,
            elapsed: Date.now() - reqStart,
            error: err.message,
          });
          engineResults[e].failed++;
        }
        
        // Progress
        if (results.length % 10 === 0 || results.length === TOTAL_REQUESTS) {
          const successful = results.filter(r => r.success).length;
          process.stdout.write(`\rProgress: ${results.length}/${TOTAL_REQUESTS} (${successful} ok)`);
        }
      }
    }
  } else {
    // Interleaved: round-robin across engines
    const promises = [];
    
    for (let i = 0; i < TOTAL_REQUESTS; i++) {
      const engineIdx = i % ENGINE_COUNT;
      const engine = engines[engineIdx];
      const audioFile = AUDIO_FILES[i % AUDIO_FILES.length];
      
      // Create promise but don't await yet
      const p = new Promise((resolve) => {
        setImmediate(() => {
          const reqStart = Date.now();
          try {
            const result = engine.transcribeFile(audioFile);
            resolve({
              engine: engineIdx,
              success: true,
              elapsed: Date.now() - reqStart,
              text: result.text.substring(0, 30),
            });
          } catch (err) {
            resolve({
              engine: engineIdx,
              success: false,
              elapsed: Date.now() - reqStart,
              error: err.message,
            });
          }
        });
      });
      
      promises.push(p);
    }
    
    // Process in batches to show progress
    const batchSize = 10;
    for (let i = 0; i < promises.length; i += batchSize) {
      const batch = promises.slice(i, i + batchSize);
      const batchResults = await Promise.all(batch);
      
      for (const r of batchResults) {
        results.push(r);
        if (r.success) {
          engineResults[r.engine].success++;
          engineResults[r.engine].totalTime += r.elapsed;
        } else {
          engineResults[r.engine].failed++;
        }
      }
      
      const successful = results.filter(r => r.success).length;
      process.stdout.write(`\rProgress: ${results.length}/${TOTAL_REQUESTS} (${successful} ok)`);
    }
  }
  
  const totalElapsed = Date.now() - startTime;
  console.log('\n');
  
  // Calculate statistics
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  const times = successful.map(r => r.elapsed).sort((a, b) => a - b);
  
  const percentile = (arr, p) => arr[Math.floor(arr.length * p / 100)] || 0;
  
  console.log('='.repeat(60));
  console.log('Overall Results');
  console.log('='.repeat(60));
  console.log(`Total requests:      ${TOTAL_REQUESTS}`);
  console.log(`Successful:          ${successful.length} (${(successful.length / TOTAL_REQUESTS * 100).toFixed(1)}%)`);
  console.log(`Failed:              ${failed.length}`);
  console.log(`Total time:          ${totalElapsed} ms`);
  console.log(`Throughput:          ${(TOTAL_REQUESTS / totalElapsed * 1000).toFixed(2)} req/s`);
  
  if (times.length > 0) {
    console.log(`\nLatency:`);
    console.log(`  Min:               ${Math.min(...times)} ms`);
    console.log(`  Max:               ${Math.max(...times)} ms`);
    console.log(`  Mean:              ${(times.reduce((a, b) => a + b, 0) / times.length).toFixed(0)} ms`);
    console.log(`  Median (p50):      ${percentile(times, 50)} ms`);
    console.log(`  p90:               ${percentile(times, 90)} ms`);
    console.log(`  p99:               ${percentile(times, 99)} ms`);
  }
  
  console.log('\nPer-Engine Results:');
  console.log('─'.repeat(50));
  console.log('Engine  Success  Failed  Avg Time');
  console.log('─'.repeat(50));
  for (let i = 0; i < ENGINE_COUNT; i++) {
    const er = engineResults[i];
    const avgTime = er.success > 0 ? (er.totalTime / er.success).toFixed(0) : 'N/A';
    console.log(`  ${i}      ${String(er.success).padStart(6)}  ${String(er.failed).padStart(6)}  ${avgTime} ms`);
  }
  
  // Check consistency - all engines should produce similar results
  console.log('\nConsistency Check:');
  const sampleResults = {};
  for (const r of successful) {
    const key = `${r.engine}-hello`;
    if (!sampleResults[r.engine] && r.text.toLowerCase().includes('hello')) {
      sampleResults[r.engine] = r.text;
    }
  }
  
  const texts = Object.values(sampleResults);
  const allSame = texts.every(t => t === texts[0]);
  if (allSame || texts.length <= 1) {
    console.log('  ✅ All engines produce consistent results');
  } else {
    console.log('  ⚠️  Engines produced different results:');
    for (const [eng, text] of Object.entries(sampleResults)) {
      console.log(`     Engine ${eng}: "${text}"`);
    }
  }
  
  if (failed.length > 0) {
    console.log(`\nFailed requests:`);
    const errorCounts = {};
    for (const f of failed) {
      const err = f.error || 'Unknown';
      errorCounts[err] = (errorCounts[err] || 0) + 1;
    }
    for (const [err, count] of Object.entries(errorCounts)) {
      console.log(`  ${count}x: ${err}`);
    }
  }
  
  // Final memory
  const finalMem = process.memoryUsage();
  console.log('\nFinal memory:');
  console.log(`  Heap: ${(finalMem.heapUsed / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  RSS:  ${(finalMem.rss / 1024 / 1024).toFixed(2)} MB`);
  
  console.log('\n' + (failed.length === 0 ? '✅ Multi-engine test passed!' : '❌ Some requests failed'));
  
  if (failed.length > TOTAL_REQUESTS * 0.1) {
    process.exit(1);
  }
}

runTest().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
