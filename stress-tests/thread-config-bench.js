#!/usr/bin/env node
/**
 * Thread Configuration Benchmark
 * 
 * Tests different worker/thread configurations to find optimal settings.
 */

const path = require('path');
const fs = require('fs');

const MODEL_PATH = './models/tiny';
const TOTAL_CHUNKS = 30;
const CHUNK_DURATION_MS = 2000;

async function testConfig(numWorkers, cpuThreadsPerWorker) {
  const { createWorkerPool } = require('../worker-pool-batcher.js');
  const { decodeAudio } = require('../index.js');
  
  // Load audio
  const files = ['./tests/fixtures/hello.wav', './tests/fixtures/numbers.wav', './tests/fixtures/sentence.wav'];
  let allSamples = [];
  for (const f of files) {
    allSamples.push(...decodeAudio(f));
  }
  const chunkSamples = Math.floor(16000 * CHUNK_DURATION_MS / 1000);
  while (allSamples.length < chunkSamples) allSamples.push(...allSamples);
  const chunkAudio = new Float32Array(allSamples.slice(0, chunkSamples));
  
  // Create pool
  const pool = createWorkerPool(MODEL_PATH, {
    numWorkers,
    cpuThreadsPerWorker,
    language: 'en',
    beamSize: 5,
  });
  
  await pool.init();
  
  // Create streams
  const streamIds = [];
  for (let i = 0; i < TOTAL_CHUNKS; i++) {
    streamIds.push(pool.createStream());
  }
  
  // Benchmark
  const start = Date.now();
  await Promise.all(streamIds.map(sid => pool.transcribeChunk(sid, chunkAudio, 0)));
  const elapsed = Date.now() - start;
  
  pool.destroy();
  
  const totalAudioS = TOTAL_CHUNKS * CHUNK_DURATION_MS / 1000;
  return {
    numWorkers,
    cpuThreadsPerWorker,
    throughput: totalAudioS / (elapsed / 1000),
    elapsedMs: elapsed,
  };
}

async function main() {
  console.log('═'.repeat(70));
  console.log('  THREAD CONFIGURATION BENCHMARK');
  console.log('═'.repeat(70));
  console.log(`  Testing ${TOTAL_CHUNKS} chunks of ${CHUNK_DURATION_MS}ms audio`);
  console.log(`  Model: ${MODEL_PATH}`);
  console.log('═'.repeat(70));
  console.log('');
  
  const results = [];
  
  // Test configurations
  const configs = [
    { workers: 1, threads: 0 },  // 1 worker, auto threads
    { workers: 2, threads: 0 },  // 2 workers, auto threads
    { workers: 4, threads: 0 },  // 4 workers, auto threads
    { workers: 4, threads: 2 },  // 4 workers, 2 threads each
    { workers: 8, threads: 0 },  // 8 workers, auto threads
    { workers: 8, threads: 1 },  // 8 workers, 1 thread each
    { workers: 10, threads: 1 }, // 10 workers, 1 thread each
  ];
  
  for (const cfg of configs) {
    process.stdout.write(`  Testing ${cfg.workers} workers, ${cfg.threads || 'auto'} threads... `);
    try {
      const result = await testConfig(cfg.workers, cfg.threads);
      results.push(result);
      console.log(`${result.throughput.toFixed(1)}x real-time (${result.elapsedMs}ms)`);
    } catch (e) {
      console.log(`ERROR: ${e.message}`);
    }
    
    // Brief pause between tests
    await new Promise(r => setTimeout(r, 1000));
  }
  
  // Find best
  const best = results.reduce((a, b) => a.throughput > b.throughput ? a : b);
  
  console.log('');
  console.log('═'.repeat(70));
  console.log('  RESULTS');
  console.log('═'.repeat(70));
  console.log('');
  console.log('  Config                    Throughput    Time');
  console.log('─'.repeat(55));
  for (const r of results) {
    const threads = r.cpuThreadsPerWorker || 'auto';
    const cfg = `${r.numWorkers} workers, ${threads} threads`;
    const isBest = r === best ? ' ⭐' : '';
    console.log(`  ${cfg.padEnd(25)} ${r.throughput.toFixed(1).padStart(6)}x      ${(r.elapsedMs/1000).toFixed(2)}s${isBest}`);
  }
  
  console.log('');
  console.log(`  Best: ${best.numWorkers} workers, ${best.cpuThreadsPerWorker || 'auto'} threads = ${best.throughput.toFixed(1)}x real-time`);
  console.log(`  Can handle ~${Math.floor(best.throughput)} concurrent 2s-chunk streams`);
  console.log('');
}

main().catch(console.error);
