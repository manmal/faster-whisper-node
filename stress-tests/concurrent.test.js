#!/usr/bin/env node
/**
 * Stress test for concurrent transcriptions
 * 
 * Run with: node stress-tests/concurrent.test.js [options]
 * 
 * Options:
 *   --requests=N     Number of concurrent requests (default: 100)
 *   --server=URL     Server URL (default: http://localhost:3000)
 *   --audio=PATH     Audio file to use (default: ./tests/fixtures/hello.wav)
 *   --batches=N      Number of batches to run (default: 1)
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// Parse command line args
const args = {};
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=');
  args[key] = value || true;
}

const CONCURRENT_REQUESTS = parseInt(args.requests || '100');
const SERVER_URL = args.server || 'http://localhost:3000';
const AUDIO_PATH = args.audio || './tests/fixtures/hello.wav';
const BATCHES = parseInt(args.batches || '1');

// Check audio file exists
if (!fs.existsSync(AUDIO_PATH)) {
  console.error('❌ Audio file not found:', AUDIO_PATH);
  process.exit(1);
}

// Load audio buffer once
const audioBuffer = fs.readFileSync(AUDIO_PATH);
console.log(`Loaded audio file: ${AUDIO_PATH} (${audioBuffer.length} bytes)`);

// Make a single transcription request
function makeRequest(id) {
  return new Promise((resolve) => {
    const startTime = Date.now();
    const url = new URL('/transcribe', SERVER_URL);
    
    const req = http.request(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': audioBuffer.length,
      },
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        const elapsed = Date.now() - startTime;
        try {
          const json = JSON.parse(data);
          resolve({
            id,
            success: json.success,
            elapsed,
            serverProcessingTime: json.processingTimeMs,
            text: json.result?.text?.substring(0, 50),
            error: json.error,
          });
        } catch (e) {
          resolve({
            id,
            success: false,
            elapsed,
            error: `Parse error: ${data.substring(0, 100)}`,
          });
        }
      });
    });
    
    req.on('error', (err) => {
      resolve({
        id,
        success: false,
        elapsed: Date.now() - startTime,
        error: err.message,
      });
    });
    
    req.write(audioBuffer);
    req.end();
  });
}

// Run a batch of concurrent requests
async function runBatch(batchNum) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Batch ${batchNum}: Starting ${CONCURRENT_REQUESTS} concurrent requests...`);
  console.log('='.repeat(60));
  
  const batchStart = Date.now();
  
  // Launch all requests simultaneously
  const promises = [];
  for (let i = 0; i < CONCURRENT_REQUESTS; i++) {
    promises.push(makeRequest(i + 1));
  }
  
  // Wait for all to complete
  const results = await Promise.all(promises);
  
  const batchElapsed = Date.now() - batchStart;
  
  // Calculate statistics
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  const times = successful.map(r => r.elapsed);
  const serverTimes = successful.map(r => r.serverProcessingTime).filter(Boolean);
  
  times.sort((a, b) => a - b);
  serverTimes.sort((a, b) => a - b);
  
  const percentile = (arr, p) => arr[Math.floor(arr.length * p / 100)] || 0;
  
  console.log(`\nResults:`);
  console.log(`  Total requests:    ${CONCURRENT_REQUESTS}`);
  console.log(`  Successful:        ${successful.length} (${(successful.length / CONCURRENT_REQUESTS * 100).toFixed(1)}%)`);
  console.log(`  Failed:            ${failed.length}`);
  console.log(`  Batch time:        ${batchElapsed} ms`);
  console.log(`  Throughput:        ${(CONCURRENT_REQUESTS / batchElapsed * 1000).toFixed(2)} req/s`);
  
  if (times.length > 0) {
    console.log(`\nClient-side latency (includes network + queuing):`);
    console.log(`  Min:               ${Math.min(...times)} ms`);
    console.log(`  Max:               ${Math.max(...times)} ms`);
    console.log(`  Mean:              ${(times.reduce((a, b) => a + b, 0) / times.length).toFixed(0)} ms`);
    console.log(`  Median (p50):      ${percentile(times, 50)} ms`);
    console.log(`  p90:               ${percentile(times, 90)} ms`);
    console.log(`  p99:               ${percentile(times, 99)} ms`);
  }
  
  if (serverTimes.length > 0) {
    console.log(`\nServer-side processing time:`);
    console.log(`  Min:               ${Math.min(...serverTimes)} ms`);
    console.log(`  Max:               ${Math.max(...serverTimes)} ms`);
    console.log(`  Mean:              ${(serverTimes.reduce((a, b) => a + b, 0) / serverTimes.length).toFixed(0)} ms`);
    console.log(`  Median (p50):      ${percentile(serverTimes, 50)} ms`);
    console.log(`  p90:               ${percentile(serverTimes, 90)} ms`);
    console.log(`  p99:               ${percentile(serverTimes, 99)} ms`);
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
  
  return { successful: successful.length, failed: failed.length, batchElapsed };
}

// Check server is running
async function checkServer() {
  return new Promise((resolve) => {
    const url = new URL('/health', SERVER_URL);
    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          resolve(json.status === 'ok');
        } catch {
          resolve(false);
        }
      });
    }).on('error', () => resolve(false));
  });
}

// Get server stats
async function getServerStats() {
  return new Promise((resolve) => {
    const url = new URL('/stats', SERVER_URL);
    http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch {
          resolve(null);
        }
      });
    }).on('error', () => resolve(null));
  });
}

// Main
async function main() {
  console.log('='.repeat(60));
  console.log('faster-whisper-rs Concurrent Stress Test');
  console.log('='.repeat(60));
  console.log(`Server:              ${SERVER_URL}`);
  console.log(`Concurrent requests: ${CONCURRENT_REQUESTS}`);
  console.log(`Batches:             ${BATCHES}`);
  console.log(`Audio file:          ${AUDIO_PATH}`);
  
  // Check server
  console.log('\nChecking server...');
  const serverOk = await checkServer();
  if (!serverOk) {
    console.error('❌ Server not responding. Start it with:');
    console.error('   node stress-tests/server.js');
    process.exit(1);
  }
  console.log('✅ Server is running');
  
  // Run batches
  let totalSuccessful = 0;
  let totalFailed = 0;
  let totalTime = 0;
  
  for (let i = 1; i <= BATCHES; i++) {
    const { successful, failed, batchElapsed } = await runBatch(i);
    totalSuccessful += successful;
    totalFailed += failed;
    totalTime += batchElapsed;
    
    // Wait between batches
    if (i < BATCHES) {
      console.log('\nWaiting 2s before next batch...');
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  
  // Final summary
  console.log(`\n${'='.repeat(60)}`);
  console.log('FINAL SUMMARY');
  console.log('='.repeat(60));
  console.log(`Total requests:      ${totalSuccessful + totalFailed}`);
  console.log(`Successful:          ${totalSuccessful}`);
  console.log(`Failed:              ${totalFailed}`);
  console.log(`Success rate:        ${(totalSuccessful / (totalSuccessful + totalFailed) * 100).toFixed(1)}%`);
  console.log(`Total time:          ${totalTime} ms`);
  console.log(`Overall throughput:  ${((totalSuccessful + totalFailed) / totalTime * 1000).toFixed(2)} req/s`);
  
  // Server stats
  const serverStats = await getServerStats();
  if (serverStats) {
    console.log(`\nServer statistics:`);
    console.log(`  Peak concurrent:   ${serverStats.peakActiveRequests}`);
    console.log(`  Avg processing:    ${serverStats.avgProcessingTimeMs} ms`);
  }
  
  console.log('\n✅ Stress test complete!');
  
  // Exit with error if too many failures
  if (totalFailed > totalSuccessful * 0.1) {
    console.error('\n❌ Too many failures (>10%)');
    process.exit(1);
  }
}

main().catch(console.error);
