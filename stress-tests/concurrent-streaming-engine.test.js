#!/usr/bin/env node
/**
 * Concurrent Streaming Engine Test
 * 
 * Tests 100 concurrent streaming sessions using the LocalAgreement algorithm.
 * Each session feeds audio in real-time chunks.
 */

const fs = require('fs');
const path = require('path');

const MODEL_PATH = './models/tiny';

// Parse args
const args = {};
for (const arg of process.argv.slice(2)) {
  const [key, value] = arg.replace(/^--/, '').split('=');
  args[key] = value || true;
}

const CONCURRENT_SESSIONS = parseInt(args.sessions || '100');
const STREAM_DURATION_S = parseFloat(args.duration || '10');
const CHUNK_SIZE_MS = parseInt(args.chunk || '1000');

async function main() {
  console.log('═'.repeat(70));
  console.log('  CONCURRENT STREAMING ENGINE TEST (LocalAgreement)');
  console.log('═'.repeat(70));
  console.log(`  Concurrent sessions: ${CONCURRENT_SESSIONS}`);
  console.log(`  Stream duration:     ${STREAM_DURATION_S}s`);
  console.log(`  Chunk size:          ${CHUNK_SIZE_MS}ms`);
  console.log(`  Model:               ${MODEL_PATH}`);
  console.log('═'.repeat(70));
  
  if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
    console.error('❌ Model not found:', MODEL_PATH);
    process.exit(1);
  }
  
  const { StreamingEngine, decodeAudio } = require('../index.js');
  
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
  
  // Create engine
  console.log('\n🔧 Creating StreamingEngine...');
  const engine = new StreamingEngine(MODEL_PATH);
  console.log(`  Sample rate: ${engine.samplingRate()}Hz`);
  
  // Create all sessions
  console.log(`\n📡 Creating ${CONCURRENT_SESSIONS} sessions...`);
  const sessions = [];
  for (let i = 0; i < CONCURRENT_SESSIONS; i++) {
    const sessionId = engine.createSession({
      minBufferSeconds: 0.5,
      stabilityMarginSeconds: 1.0,
      contextOverlapSeconds: 0.3,
      language: 'en',
      beamSize: 3,
    });
    sessions.push({
      id: sessionId,
      stableCount: 0,
      previewUpdates: 0,
      processingTimeMs: 0,
      chunksProcessed: 0,
    });
  }
  console.log(`  Sessions created: ${sessions.length}`);
  
  // Feed audio in chunks
  const chunkSamples = Math.floor(16000 * CHUNK_SIZE_MS / 1000);
  const totalChunks = Math.ceil(allSamples.length / chunkSamples);
  
  console.log(`\n🎤 Streaming ${totalChunks} chunks to ${CONCURRENT_SESSIONS} sessions...`);
  console.log(`  Total operations: ${totalChunks * CONCURRENT_SESSIONS}`);
  
  const testStart = Date.now();
  let totalOperations = 0;
  let totalStableSegments = 0;
  
  // Process each chunk for all sessions
  for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
    const start = chunkIdx * chunkSamples;
    const end = Math.min((chunkIdx + 1) * chunkSamples, allSamples.length);
    const chunk = allSamples.slice(start, end);
    
    // Process this chunk for all sessions
    const chunkStart = Date.now();
    
    for (const session of sessions) {
      const opStart = Date.now();
      const result = engine.processAudio(session.id, chunk);
      const opTime = Date.now() - opStart;
      
      session.processingTimeMs += opTime;
      session.chunksProcessed++;
      session.stableCount += (result.stableSegments?.length || 0);
      if (result.previewText) session.previewUpdates++;
      
      totalStableSegments += (result.stableSegments?.length || 0);
      totalOperations++;
    }
    
    const chunkTime = Date.now() - chunkStart;
    const elapsed = Date.now() - testStart;
    const progress = ((chunkIdx + 1) / totalChunks * 100).toFixed(0);
    
    process.stdout.write(`\r  [${(elapsed/1000).toFixed(1)}s] Chunk ${chunkIdx + 1}/${totalChunks} (${progress}%) | ${chunkTime}ms for ${CONCURRENT_SESSIONS} sessions`);
  }
  
  // Flush all sessions
  console.log('\n\n🔚 Flushing all sessions...');
  const flushStart = Date.now();
  
  for (const session of sessions) {
    const result = engine.flushSession(session.id);
    session.stableCount += (result.stableSegments?.length || 0);
    totalStableSegments += (result.stableSegments?.length || 0);
    engine.closeSession(session.id);
  }
  
  const flushTime = Date.now() - flushStart;
  const totalTime = Date.now() - testStart;
  
  // Stats
  console.log('\n' + '═'.repeat(70));
  console.log('  RESULTS');
  console.log('═'.repeat(70));
  
  const totalAudioS = CONCURRENT_SESSIONS * STREAM_DURATION_S;
  const throughput = totalAudioS / (totalTime / 1000);
  
  const avgProcessingPerOp = sessions.reduce((sum, s) => sum + s.processingTimeMs, 0) / totalOperations;
  const avgStablePerSession = sessions.reduce((sum, s) => sum + s.stableCount, 0) / sessions.length;
  
  console.log(`\n📊 Performance:`);
  console.log(`  Total time:            ${(totalTime / 1000).toFixed(2)}s`);
  console.log(`  Total audio:           ${totalAudioS.toFixed(0)}s (${CONCURRENT_SESSIONS} x ${STREAM_DURATION_S}s)`);
  console.log(`  Throughput:            ${throughput.toFixed(1)}x real-time`);
  console.log(`  Avg per operation:     ${avgProcessingPerOp.toFixed(0)}ms`);
  console.log(`  Flush time:            ${flushTime}ms`);
  
  console.log(`\n📈 Streaming Quality:`);
  console.log(`  Total stable segments: ${totalStableSegments}`);
  console.log(`  Avg per session:       ${avgStablePerSession.toFixed(1)} segments`);
  console.log(`  Active sessions:       ${engine.sessionCount()}`);
  
  // Check if we can keep up
  const chunksPerSecondNeeded = CONCURRENT_SESSIONS / (CHUNK_SIZE_MS / 1000);
  const actualChunksPerSecond = totalOperations / (totalTime / 1000);
  const canKeepUp = actualChunksPerSecond >= chunksPerSecondNeeded;
  
  console.log(`\n🎯 Real-Time Capability:`);
  console.log(`  Chunks needed/sec:     ${chunksPerSecondNeeded.toFixed(1)}`);
  console.log(`  Actual chunks/sec:     ${actualChunksPerSecond.toFixed(1)}`);
  console.log(`  Can keep up:           ${canKeepUp ? '✅ YES' : '❌ NO'}`);
  
  console.log('\n' + '═'.repeat(70));
  
  if (!canKeepUp) {
    console.log(`\n⚠️  System cannot keep up with ${CONCURRENT_SESSIONS} real-time streams.`);
    console.log(`   Max sustainable streams: ~${Math.floor(actualChunksPerSecond * CHUNK_SIZE_MS / 1000)}`);
  } else {
    console.log(`\n🎉 System CAN handle ${CONCURRENT_SESSIONS} concurrent real-time streams!`);
  }
  
  console.log('');
}

main().catch(console.error);
