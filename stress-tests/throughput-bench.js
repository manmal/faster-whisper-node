#!/usr/bin/env node
/**
 * Simple Throughput Benchmark
 * 
 * Measures maximum throughput for concurrent transcription.
 */

const path = require('path');
const fs = require('fs');

// Parse args
const args = {};
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=');
  args[key] = value || true;
}

const TOTAL_CHUNKS = parseInt(args.chunks || '100');
const NUM_WORKERS = parseInt(args.workers || '8');
const CHUNK_DURATION_MS = parseInt(args.chunk || '2000');
const MODEL_PATH = args.model || './models/tiny';

async function main() {
  console.log('═'.repeat(60));
  console.log('  THROUGHPUT BENCHMARK');
  console.log('═'.repeat(60));
  console.log(`  Total chunks:    ${TOTAL_CHUNKS}`);
  console.log(`  Workers:         ${NUM_WORKERS}`);
  console.log(`  Chunk duration:  ${CHUNK_DURATION_MS}ms`);
  console.log(`  Model:           ${MODEL_PATH}`);
  console.log('═'.repeat(60));
  
  const { WorkerPoolBatcher, createWorkerPool } = require('../worker-pool-batcher.js');
  const { decodeAudio } = require('../index.js');
  
  // Load audio from multiple files to get enough length
  const files = ['./tests/fixtures/hello.wav', './tests/fixtures/numbers.wav', './tests/fixtures/sentence.wav'];
  let allSamples = [];
  for (const f of files) {
    const s = decodeAudio(f);
    allSamples.push(...s);
  }
  
  const chunkSamples = Math.floor(16000 * CHUNK_DURATION_MS / 1000);
  // Extend if needed
  while (allSamples.length < chunkSamples) {
    allSamples.push(...allSamples);
  }
  const chunkAudio = new Float32Array(allSamples.slice(0, chunkSamples));
  
  console.log(`\n📁 Loaded ${(chunkAudio.length/16000).toFixed(2)}s audio chunk`);
  
  // Create pool
  console.log(`\n🔧 Creating worker pool...`);
  const pool = createWorkerPool(MODEL_PATH, {
    numWorkers: NUM_WORKERS,
    language: 'en',
    beamSize: 5,
    wordTimestamps: false,
  });
  
  await pool.init();
  console.log(`  All ${NUM_WORKERS} workers ready`);
  
  // Pre-create streams for all chunks
  const streamIds = [];
  for (let i = 0; i < TOTAL_CHUNKS; i++) {
    streamIds.push(pool.createStream());
  }
  
  // Benchmark 1: All at once
  console.log(`\n🚀 Benchmark 1: Submit all ${TOTAL_CHUNKS} chunks at once...`);
  const allAtOnceStart = Date.now();
  
  const allPromises = streamIds.map((streamId, i) => 
    pool.transcribeChunk(streamId, chunkAudio, 0)
      .then(result => ({
        i,
        processingTimeMs: result.processingTimeMs,
        queueTimeMs: result.queueTimeMs,
      }))
  );
  
  const allResults = await Promise.all(allPromises);
  const allAtOnceTime = Date.now() - allAtOnceStart;
  
  const totalAudioS = TOTAL_CHUNKS * CHUNK_DURATION_MS / 1000;
  const throughput = totalAudioS / (allAtOnceTime / 1000);
  
  console.log(`\n📊 Results:`);
  console.log(`  Total time:      ${(allAtOnceTime/1000).toFixed(2)}s`);
  console.log(`  Total audio:     ${totalAudioS.toFixed(1)}s`);
  console.log(`  Throughput:      ${throughput.toFixed(1)}x real-time`);
  console.log(`  Chunks/second:   ${(TOTAL_CHUNKS / (allAtOnceTime/1000)).toFixed(1)}`);
  
  const processingTimes = allResults.map(r => r.processingTimeMs);
  const queueTimes = allResults.map(r => r.queueTimeMs);
  
  console.log(`\n  Processing time:`);
  console.log(`    Min: ${Math.min(...processingTimes)}ms, Max: ${Math.max(...processingTimes)}ms`);
  console.log(`    Avg: ${(processingTimes.reduce((a,b)=>a+b,0)/processingTimes.length).toFixed(0)}ms`);
  
  console.log(`\n  Queue time:`);
  console.log(`    Min: ${Math.min(...queueTimes)}ms, Max: ${Math.max(...queueTimes)}ms`);
  console.log(`    Avg: ${(queueTimes.reduce((a,b)=>a+b,0)/queueTimes.length).toFixed(0)}ms`);
  
  // Calculate if we can handle 100 concurrent streams
  const streamsWeCanHandle = throughput / 2 * 2; // chunks per second * seconds per chunk
  console.log(`\n📈 Capacity estimate:`);
  console.log(`  Can handle ~${streamsWeCanHandle.toFixed(0)} concurrent ${CHUNK_DURATION_MS}ms streams`);
  
  // Cleanup
  pool.destroy();
  
  console.log('\n' + '═'.repeat(60));
}

main().catch(console.error);
