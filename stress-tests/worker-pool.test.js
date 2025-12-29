#!/usr/bin/env node
/**
 * Worker Pool Stress Test
 * 
 * Tests true parallelism using Node.js worker threads.
 * Each worker has its own Engine instance for maximum throughput.
 * 
 * Run with: node stress-tests/worker-pool.test.js [options]
 * 
 * Options:
 *   --requests=N     Total requests (default: 100)
 *   --workers=N      Worker pool size (default: CPU count)
 *   --audio=PATH     Audio file (default: ./tests/fixtures/hello.wav)
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

const TOTAL_REQUESTS = parseInt(args.requests || '100');
const WORKER_COUNT = parseInt(args.workers || os.cpus().length);
const AUDIO_PATH = args.audio || './tests/fixtures/hello.wav';
const MODEL_PATH = args.model || './models/tiny';

if (!isMainThread) {
  // Worker thread code
  const { Engine } = require(workerData.enginePath);
  const engine = new Engine(workerData.modelPath);
  
  parentPort.on('message', (msg) => {
    if (msg.type === 'transcribe') {
      const start = Date.now();
      try {
        const result = engine.transcribeFile(msg.audioPath, msg.options || {});
        parentPort.postMessage({
          type: 'result',
          id: msg.id,
          success: true,
          elapsed: Date.now() - start,
          text: result.text.substring(0, 50),
          segments: result.segments.length,
        });
      } catch (err) {
        parentPort.postMessage({
          type: 'result',
          id: msg.id,
          success: false,
          elapsed: Date.now() - start,
          error: err.message,
        });
      }
    } else if (msg.type === 'exit') {
      process.exit(0);
    }
  });
  
  parentPort.postMessage({ type: 'ready' });
} else {
  // Main thread
  async function main() {
    console.log('='.repeat(60));
    console.log('Worker Pool Stress Test');
    console.log('='.repeat(60));
    console.log(`Total requests:  ${TOTAL_REQUESTS}`);
    console.log(`Worker threads:  ${WORKER_COUNT}`);
    console.log(`Model:           ${MODEL_PATH}`);
    console.log(`Audio:           ${AUDIO_PATH}`);
    
    // Verify files exist
    if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
      console.error('❌ Model not found:', MODEL_PATH);
      process.exit(1);
    }
    if (!fs.existsSync(AUDIO_PATH)) {
      console.error('❌ Audio file not found:', AUDIO_PATH);
      process.exit(1);
    }
    
    console.log('\nInitializing workers...');
    
    // Create worker pool
    const workers = [];
    const workerStatus = new Map();
    const results = [];
    let completedRequests = 0;
    let nextRequestId = 0;
    
    const enginePath = path.resolve(__dirname, '../index.js');
    const modelPathAbs = path.resolve(MODEL_PATH);
    const audioPathAbs = path.resolve(AUDIO_PATH);
    
    await new Promise((resolve, reject) => {
      let readyWorkers = 0;
      
      for (let i = 0; i < WORKER_COUNT; i++) {
        const worker = new Worker(__filename, {
          workerData: { 
            enginePath, 
            modelPath: modelPathAbs,
          },
        });
        
        worker.on('message', (msg) => {
          if (msg.type === 'ready') {
            workerStatus.set(i, 'idle');
            readyWorkers++;
            console.log(`  Worker ${i + 1}/${WORKER_COUNT} ready`);
            if (readyWorkers === WORKER_COUNT) {
              resolve();
            }
          } else if (msg.type === 'result') {
            results.push(msg);
            completedRequests++;
            workerStatus.set(i, 'idle');
            
            // Progress
            if (completedRequests % 10 === 0 || completedRequests === TOTAL_REQUESTS) {
              const successful = results.filter(r => r.success).length;
              process.stdout.write(`\rProgress: ${completedRequests}/${TOTAL_REQUESTS} (${successful} ok)`);
            }
            
            // Send next request if available
            if (nextRequestId < TOTAL_REQUESTS) {
              workerStatus.set(i, 'busy');
              worker.postMessage({
                type: 'transcribe',
                id: nextRequestId++,
                audioPath: audioPathAbs,
              });
            }
          }
        });
        
        worker.on('error', (err) => {
          console.error(`Worker ${i} error:`, err);
          reject(err);
        });
        
        workers.push(worker);
      }
    });
    
    console.log(`\nStarting ${TOTAL_REQUESTS} requests across ${WORKER_COUNT} workers...\n`);
    
    const startTime = Date.now();
    
    // Initial dispatch - fill all workers
    for (let i = 0; i < WORKER_COUNT && nextRequestId < TOTAL_REQUESTS; i++) {
      workerStatus.set(i, 'busy');
      workers[i].postMessage({
        type: 'transcribe',
        id: nextRequestId++,
        audioPath: audioPathAbs,
      });
    }
    
    // Wait for completion
    await new Promise((resolve) => {
      const checkComplete = setInterval(() => {
        if (completedRequests >= TOTAL_REQUESTS) {
          clearInterval(checkComplete);
          resolve();
        }
      }, 100);
    });
    
    const elapsed = Date.now() - startTime;
    console.log('\n');
    
    // Terminate workers
    for (const worker of workers) {
      worker.postMessage({ type: 'exit' });
    }
    
    // Calculate statistics
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);
    const times = successful.map(r => r.elapsed).sort((a, b) => a - b);
    
    const percentile = (arr, p) => arr[Math.floor(arr.length * p / 100)] || 0;
    
    console.log('='.repeat(60));
    console.log('Results');
    console.log('='.repeat(60));
    console.log(`Total requests:      ${TOTAL_REQUESTS}`);
    console.log(`Successful:          ${successful.length} (${(successful.length / TOTAL_REQUESTS * 100).toFixed(1)}%)`);
    console.log(`Failed:              ${failed.length}`);
    console.log(`Total time:          ${elapsed} ms`);
    console.log(`Throughput:          ${(TOTAL_REQUESTS / elapsed * 1000).toFixed(2)} req/s`);
    
    if (times.length > 0) {
      console.log(`\nLatency:`);
      console.log(`  Min:               ${Math.min(...times)} ms`);
      console.log(`  Max:               ${Math.max(...times)} ms`);
      console.log(`  Mean:              ${(times.reduce((a, b) => a + b, 0) / times.length).toFixed(0)} ms`);
      console.log(`  Median (p50):      ${percentile(times, 50)} ms`);
      console.log(`  p90:               ${percentile(times, 90)} ms`);
      console.log(`  p99:               ${percentile(times, 99)} ms`);
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
    
    console.log('\n✅ Worker pool test complete!');
    
    if (failed.length > TOTAL_REQUESTS * 0.1) {
      console.error('❌ Too many failures (>10%)');
      process.exit(1);
    }
  }
  
  main().catch((err) => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}
