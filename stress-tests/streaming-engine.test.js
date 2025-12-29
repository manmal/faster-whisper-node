#!/usr/bin/env node
/**
 * StreamingEngine Test - LocalAgreement Algorithm
 * 
 * Tests the true streaming transcription using the LocalAgreement algorithm.
 * Audio is fed in chunks, and the engine returns stable (final) segments
 * plus preview text that may change.
 */

const fs = require('fs');
const path = require('path');

const MODEL_PATH = './models/tiny';

async function main() {
  console.log('═'.repeat(70));
  console.log('  STREAMING ENGINE TEST (LocalAgreement Algorithm)');
  console.log('═'.repeat(70));
  
  // Check model
  if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
    console.error('❌ Model not found:', MODEL_PATH);
    process.exit(1);
  }
  
  const { StreamingEngine, decodeAudio } = require('../index.js');
  
  // Load audio
  console.log('\n📁 Loading audio files...');
  const files = [
    './tests/fixtures/hello.wav',
    './tests/fixtures/numbers.wav',
    './tests/fixtures/sentence.wav',
  ].filter(f => fs.existsSync(f));
  
  let allSamples = [];
  for (const f of files) {
    const samples = decodeAudio(f);
    allSamples.push(...samples);
  }
  console.log(`  Total audio: ${(allSamples.length / 16000).toFixed(2)}s`);
  
  // Create streaming engine
  console.log('\n🔧 Creating StreamingEngine...');
  const engine = new StreamingEngine(MODEL_PATH);
  console.log(`  Sampling rate: ${engine.samplingRate()}Hz`);
  
  // Create a session
  console.log('\n📡 Creating streaming session...');
  const sessionId = engine.createSession({
    minBufferSeconds: 1.0,
    stabilityMarginSeconds: 1.5,
    contextOverlapSeconds: 0.5,
    language: 'en',
    beamSize: 5,
  });
  console.log(`  Session ID: ${sessionId}`);
  
  // Feed audio in chunks (simulate real-time streaming)
  const chunkSizeMs = 500; // 500ms chunks
  const chunkSamples = Math.floor(16000 * chunkSizeMs / 1000);
  const totalChunks = Math.ceil(allSamples.length / chunkSamples);
  
  console.log(`\n🎤 Feeding ${totalChunks} chunks (${chunkSizeMs}ms each)...`);
  console.log('─'.repeat(70));
  
  let totalStableSegments = 0;
  let fullStableText = '';
  let lastPreview = '';
  
  for (let i = 0; i < totalChunks; i++) {
    const start = i * chunkSamples;
    const end = Math.min((i + 1) * chunkSamples, allSamples.length);
    const chunk = allSamples.slice(start, end);
    
    const audioTimeS = (i + 1) * chunkSizeMs / 1000;
    const chunkDuration = chunk.length / 16000;
    
    // Simulate real-time delay (optional, comment out for speed test)
    // await new Promise(r => setTimeout(r, chunkSizeMs));
    
    // Process the chunk
    const processStart = Date.now();
    const result = engine.processAudio(sessionId, chunk);
    const processTime = Date.now() - processStart;
    
    // Report results
    if (result.stableSegments && result.stableSegments.length > 0) {
      for (const seg of result.stableSegments) {
        console.log(`  ✅ FINAL [${seg.start.toFixed(2)}s-${seg.end.toFixed(2)}s]: "${seg.text}"`);
        fullStableText += seg.text;
        totalStableSegments++;
      }
    }
    
    if (result.previewText && result.previewText !== lastPreview) {
      console.log(`  ⏳ PREVIEW: "${result.previewText}" (buffer: ${result.bufferDuration.toFixed(2)}s)`);
      lastPreview = result.previewText;
    }
    
    // Show progress every 10 chunks
    if ((i + 1) % 10 === 0) {
      console.log(`  ... processed ${i + 1}/${totalChunks} chunks, ${processTime}ms/chunk`);
    }
  }
  
  // Flush remaining audio
  console.log('\n🔚 Flushing session...');
  const flushResult = engine.flushSession(sessionId);
  
  if (flushResult.stableSegments && flushResult.stableSegments.length > 0) {
    for (const seg of flushResult.stableSegments) {
      console.log(`  ✅ FINAL [${seg.start.toFixed(2)}s-${seg.end.toFixed(2)}s]: "${seg.text}"`);
      fullStableText += seg.text;
      totalStableSegments++;
    }
  }
  
  // Close session
  engine.closeSession(sessionId);
  
  // Summary
  console.log('\n' + '═'.repeat(70));
  console.log('  RESULTS');
  console.log('═'.repeat(70));
  console.log(`  Total stable segments: ${totalStableSegments}`);
  console.log(`  Final transcription: "${fullStableText.trim()}"`);
  console.log(`  Audio duration: ${(allSamples.length / 16000).toFixed(2)}s`);
  console.log('═'.repeat(70));
}

main().catch(console.error);
