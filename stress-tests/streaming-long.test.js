#!/usr/bin/env node
/**
 * Streaming Long Audio Test
 * 
 * Tests the LocalAgreement algorithm with longer audio to verify
 * stable segments are properly emitted during streaming.
 */

const fs = require('fs');
const path = require('path');

const MODEL_PATH = './models/tiny';

async function main() {
  console.log('═'.repeat(70));
  console.log('  STREAMING LONG AUDIO TEST');
  console.log('═'.repeat(70));
  
  if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
    console.error('❌ Model not found:', MODEL_PATH);
    process.exit(1);
  }
  
  const { StreamingEngine, decodeAudio } = require('../index.js');
  
  // Load audio and repeat to get ~30 seconds
  console.log('\n📁 Loading and extending audio...');
  const files = [
    './tests/fixtures/hello.wav',
    './tests/fixtures/numbers.wav',
    './tests/fixtures/sentence.wav',
  ].filter(f => fs.existsSync(f));
  
  let baseSamples = [];
  for (const f of files) {
    const samples = decodeAudio(f);
    baseSamples.push(...samples);
  }
  
  // Extend to ~20 seconds
  const targetDuration = 20;
  let allSamples = [];
  while (allSamples.length < targetDuration * 16000) {
    allSamples.push(...baseSamples);
  }
  allSamples = allSamples.slice(0, targetDuration * 16000);
  
  console.log(`  Audio duration: ${(allSamples.length / 16000).toFixed(2)}s`);
  
  // Create streaming engine
  console.log('\n🔧 Creating StreamingEngine...');
  const engine = new StreamingEngine(MODEL_PATH);
  
  // Create session with custom settings
  const sessionId = engine.createSession({
    minBufferSeconds: 1.0,
    stabilityMarginSeconds: 1.0, // Reduced for faster feedback
    contextOverlapSeconds: 0.3,
    language: 'en',
    beamSize: 3, // Faster
  });
  
  console.log(`  Session ID: ${sessionId}`);
  
  // Feed audio in 1-second chunks
  const chunkSizeMs = 1000;
  const chunkSamples = Math.floor(16000 * chunkSizeMs / 1000);
  const totalChunks = Math.ceil(allSamples.length / chunkSamples);
  
  console.log(`\n🎤 Streaming ${totalChunks} chunks (${chunkSizeMs}ms each)...`);
  console.log('─'.repeat(70));
  
  let totalStableSegments = 0;
  let stableTextParts = [];
  let lastPreview = '';
  let totalProcessingMs = 0;
  
  for (let i = 0; i < totalChunks; i++) {
    const start = i * chunkSamples;
    const end = Math.min((i + 1) * chunkSamples, allSamples.length);
    const chunk = allSamples.slice(start, end);
    
    const audioTimeS = (i + 1) * chunkSizeMs / 1000;
    
    // Process the chunk
    const processStart = Date.now();
    const result = engine.processAudio(sessionId, chunk);
    const processTime = Date.now() - processStart;
    totalProcessingMs += processTime;
    
    // Report stable segments immediately
    if (result.stableSegments && result.stableSegments.length > 0) {
      for (const seg of result.stableSegments) {
        console.log(`  [${audioTimeS.toFixed(1)}s] ✅ FINAL: "${seg.text.trim()}"`);
        stableTextParts.push(seg.text);
        totalStableSegments++;
      }
    }
    
    // Only show preview if it changed significantly
    const preview = result.previewText || '';
    if (preview && preview !== lastPreview) {
      // Only show first 50 chars of preview
      const shortPreview = preview.length > 50 ? preview.substring(0, 50) + '...' : preview;
      console.log(`  [${audioTimeS.toFixed(1)}s] ⏳ preview: "${shortPreview}"`);
      lastPreview = preview;
    }
  }
  
  // Flush
  console.log('\n🔚 Flushing remaining audio...');
  const flushResult = engine.flushSession(sessionId);
  
  if (flushResult.stableSegments && flushResult.stableSegments.length > 0) {
    for (const seg of flushResult.stableSegments) {
      console.log(`  ✅ FINAL (flush): "${seg.text.trim()}"`);
      stableTextParts.push(seg.text);
      totalStableSegments++;
    }
  }
  
  engine.closeSession(sessionId);
  
  // Stats
  const avgProcessingMs = totalProcessingMs / totalChunks;
  const rtf = avgProcessingMs / chunkSizeMs;
  
  console.log('\n' + '═'.repeat(70));
  console.log('  RESULTS');
  console.log('═'.repeat(70));
  console.log(`  Total stable segments: ${totalStableSegments}`);
  console.log(`  Avg processing time: ${avgProcessingMs.toFixed(0)}ms per ${chunkSizeMs}ms chunk`);
  console.log(`  Real-time factor: ${rtf.toFixed(2)} (< 1.0 = faster than real-time)`);
  console.log(`\n  Full transcription:`);
  console.log(`  "${stableTextParts.join('').trim()}"`);
  console.log('═'.repeat(70));
}

main().catch(console.error);
