#!/usr/bin/env node
/**
 * Streaming Worker Pool Test
 * 
 * Tests parallel streaming using multiple workers, each with LocalAgreement.
 */

const fs = require('fs');
const path = require('path');

const MODEL_PATH = './models/tiny';

const args = {};
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=');
  args[key] = value || true;
}

const CONCURRENT_SESSIONS = parseInt(args.sessions || '100');
const STREAM_DURATION_S = parseFloat(args.duration || '10');
const CHUNK_SIZE_MS = parseInt(args.chunk || '1000');
const NUM_WORKERS = parseInt(args.workers || '8');

async function main() {
  console.log('═'.repeat(70));
  console.log('  STREAMING WORKER POOL TEST (Parallel LocalAgreement)');
  console.log('═'.repeat(70));
  console.log(`  Concurrent sessions: ${CONCURRENT_SESSIONS}`);
  console.log(`  Stream duration:     ${STREAM_DURATION_S}s`);
  console.log(`  Chunk size:          ${CHUNK_SIZE_MS}ms`);
  console.log(`  Workers:             ${NUM_WORKERS}`);
  console.log(`  Model:               ${MODEL_PATH}`);
  console.log('═'.repeat(70));
  
  if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
    console.error('❌ Model not found:', MODEL_PATH);
    process.exit(1);
  }
  
  const { createStreamingPool } = require('../streaming-worker-pool.js');
  const { decodeAudio } = require('../index.js');
  
  // Load and extend audio
  console.log('\n📁 Loading audio...');
  const files = [
    './tests/fixtures/hello.wav',
    './tests/fixtures/numbers.wav',
    './tests/fixtures/sentence.wav',
  ].filter(f => fs.existsSync(f));
  
  let baseSamples = [];
  for (const f of files) {
    baseSamples.push(...decodeAudio(f));
  }
  
  const targetSamples = Math.ceil(STREAM_DURATION_S * 16000);
  let allSamples = [];
  while (allSamples.length < targetSamples) {
    allSamples.push(...baseSamples);
  }
  allSamples = allSamples.slice(0, targetSamples);
  console.log(`  Audio: ${(allSamples.length / 16000).toFixed(2)}s per session`);
  
  // Create worker pool
  console.log(`\n🔧 Creating StreamingWorkerPool with ${NUM_WORKERS} workers...`);
  const pool = createStreamingPool(MODEL_PATH, { numWorkers: NUM_WORKERS });
  await pool.init();
  console.log(`  Workers ready: ${pool.workers.length}`);
  
  // Create all sessions
  console.log(`\n📡 Creating ${CONCURRENT_SESSIONS} sessions...`);
  const sessions = [];
  for (let i = 0; i < CONCURRENT_SESSIONS; i++) {
    const sessionId = await pool.createSession({
      minBufferSeconds: 0.5,
      stabilityMarginSeconds: 1.0,
      contextOverlapSeconds: 0.3,
      language: 'en',
      beamSize: 3,
    });
    sessions.push({
      id: sessionId,
      stableCount: 0,
      processingTimeMs: 0,
    });
  }
  console.log(`  Sessions created: ${sessions.length}`);
  
  // Feed audio in chunks
  const chunkSamples = Math.floor(16000 * CHUNK_SIZE_MS / 1000);
  const totalChunks = Math.ceil(allSamples.length / chunkSamples);
  
  console.log(`\n🎤 Streaming ${totalChunks} chunks to ${CONCURRENT_SESSIONS} sessions...`);
  
  const testStart = Date.now();
  let totalStableSegments = 0;
  
  // Process chunks in parallel across all sessions
  for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
    const start = chunkIdx * chunkSamples;
    const end = Math.min((chunkIdx + 1) * chunkSamples, allSamples.length);
    const chunk = allSamples.slice(start, end);
    
    const chunkStart = Date.now();
    
    // Process all sessions in parallel
    const promises = sessions.map(async (session) => {
      const opStart = Date.now();
      const result = await pool.processAudio(session.id, chunk);
      session.processingTimeMs += Date.now() - opStart;
      session.stableCount += (result.stableSegments?.length || 0);
      return result.stableSegments?.length || 0;
    });
    
    const results = await Promise.all(promises);
    totalStableSegments += results.reduce((a, b) => a + b, 0);
    
    const chunkTime = Date.now() - chunkStart;
    const elapsed = Date.now() - testStart;
    const progress = ((chunkIdx + 1) / totalChunks * 100).toFixed(0);
    
    process.stdout.write(`\r  [${(elapsed/1000).toFixed(1)}s] Chunk ${chunkIdx + 1}/${totalChunks} (${progress}%) | ${chunkTime}ms for ${CONCURRENT_SESSIONS} sessions`);
  }
  
  // Flush all sessions
  console.log('\n\n🔚 Flushing all sessions...');
  const flushStart = Date.now();
  
  const flushPromises = sessions.map(async (session) => {
    const result = await pool.flushSession(session.id);
    await pool.closeSession(session.id);
    return result.stableSegments?.length || 0;
  });
  
  const flushResults = await Promise.all(flushPromises);
  totalStableSegments += flushResults.reduce((a, b) => a + b, 0);
  
  const flushTime = Date.now() - flushStart;
  const totalTime = Date.now() - testStart;
  
  // Cleanup
  pool.destroy();
  
  // Stats
  console.log('\n' + '═'.repeat(70));
  console.log('  RESULTS');
  console.log('═'.repeat(70));
  
  const totalAudioS = CONCURRENT_SESSIONS * STREAM_DURATION_S;
  const throughput = totalAudioS / (totalTime / 1000);
  const totalOperations = totalChunks * CONCURRENT_SESSIONS;
  const avgProcessingPerSession = sessions.reduce((sum, s) => sum + s.processingTimeMs, 0) / sessions.length;
  
  console.log(`\n📊 Performance:`);
  console.log(`  Total time:            ${(totalTime / 1000).toFixed(2)}s`);
  console.log(`  Total audio:           ${totalAudioS.toFixed(0)}s (${CONCURRENT_SESSIONS} x ${STREAM_DURATION_S}s)`);
  console.log(`  Throughput:            ${throughput.toFixed(1)}x real-time`);
  console.log(`  Avg per session:       ${avgProcessingPerSession.toFixed(0)}ms total`);
  console.log(`  Flush time:            ${flushTime}ms`);
  
  console.log(`\n📈 Streaming Quality:`);
  console.log(`  Total stable segments: ${totalStableSegments}`);
  console.log(`  Avg per session:       ${(totalStableSegments / CONCURRENT_SESSIONS).toFixed(1)} segments`);
  
  // Real-time check
  const chunksPerSecondNeeded = CONCURRENT_SESSIONS / (CHUNK_SIZE_MS / 1000);
  const actualChunksPerSecond = totalOperations / (totalTime / 1000);
  const canKeepUp = actualChunksPerSecond >= chunksPerSecondNeeded * 0.9; // Allow 10% slack
  
  console.log(`\n🎯 Real-Time Capability:`);
  console.log(`  Chunks needed/sec:     ${chunksPerSecondNeeded.toFixed(1)}`);
  console.log(`  Actual chunks/sec:     ${actualChunksPerSecond.toFixed(1)}`);
  console.log(`  Can keep up:           ${canKeepUp ? '✅ YES' : '❌ NO'}`);
  
  // Capacity estimate
  const maxStreams = Math.floor(actualChunksPerSecond * CHUNK_SIZE_MS / 1000);
  
  console.log(`\n🔮 Capacity Estimate:`);
  console.log(`  Max concurrent streams: ~${maxStreams}`);
  console.log(`  Workers efficiency:     ${(throughput / NUM_WORKERS * 100 / STREAM_DURATION_S).toFixed(1)}%`);
  
  console.log('\n' + '═'.repeat(70));
  
  if (canKeepUp) {
    console.log(`\n🎉 SUCCESS: System CAN handle ${CONCURRENT_SESSIONS} concurrent real-time streams!`);
  } else {
    console.log(`\n⚠️  System cannot keep up with ${CONCURRENT_SESSIONS} real-time streams.`);
    console.log(`   Recommended: Reduce to ~${maxStreams} streams or add more workers.`);
  }
  
  console.log('');
}

main().catch(console.error);
