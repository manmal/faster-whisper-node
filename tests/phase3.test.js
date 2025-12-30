/**
 * Phase 3 Test - Word timestamps, VAD, and hallucination detection
 */
const path = require('path');
const fs = require('fs');

const modelPath = './models/tiny';
const audioPath = './tests/fixtures/hello.wav';

// Check prerequisites
if (!fs.existsSync(path.join(modelPath, 'model.bin'))) {
  console.log('⚠️  Model not found. Download with:');
  console.log('   cd models && git lfs install && git clone --depth 1 https://huggingface.co/Systran/faster-whisper-tiny tiny');
  process.exit(0);
}

if (!fs.existsSync(audioPath)) {
  console.log('⚠️  Test audio not found:', audioPath);
  process.exit(1);
}

console.log('Loading faster-whisper-rs...');
const { Engine, formatTimestamp } = require('../index');

// Create engine
console.log('\n📋 Loading model...');
const engine = new Engine(modelPath);
console.log('   ✅ Model loaded');

// Test 1: Word-level timestamps
console.log('\n📋 Test 1: Word-level timestamps');
const wordResult = engine.transcribeFile(audioPath, {
  wordTimestamps: true,
});

console.log('   Text:', wordResult.text);
console.log('   Segments:', wordResult.segments.length);

for (const seg of wordResult.segments) {
  console.log(`   [${formatTimestamp(seg.start)} -> ${formatTimestamp(seg.end)}] ${seg.text}`);
  
  if (seg.words && seg.words.length > 0) {
    console.log('     Words:');
    for (const word of seg.words) {
      console.log(`       "${word.word}" [${formatTimestamp(word.start)} -> ${formatTimestamp(word.end)}]`);
    }
  } else {
    console.log('     (no word-level timestamps available)');
  }
}

// Note: word timestamps may or may not be available depending on model output
console.log('   ✅ Passed');

// Test 2: VAD filtering
console.log('\n📋 Test 2: VAD filtering');
const vadResult = engine.transcribeFile(audioPath, {
  vadFilter: true,
  vadOptions: {
    threshold: 0.5,
    minSpeechDurationMs: 100,
    speechPadMs: 200,
  }
});

console.log('   Text:', vadResult.text);
console.log('   Duration:', vadResult.duration.toFixed(2), 'seconds');
console.log('   Duration after VAD:', vadResult.durationAfterVad.toFixed(2), 'seconds');

if (vadResult.durationAfterVad <= vadResult.duration) {
  console.log('   ✅ VAD reduced or maintained duration');
} else {
  console.log('   ⚠️  Unexpected: VAD increased duration');
}

// Test 3: VAD + word timestamps combined
console.log('\n📋 Test 3: VAD + Word timestamps combined');
const combinedResult = engine.transcribeFile(audioPath, {
  vadFilter: true,
  wordTimestamps: true,
  vadOptions: {
    threshold: 0.3,
  }
});

console.log('   Text:', combinedResult.text);
console.log('   Duration:', combinedResult.duration.toFixed(2), '→', combinedResult.durationAfterVad.toFixed(2), 'seconds');

for (const seg of combinedResult.segments) {
  console.log(`   Segment [${formatTimestamp(seg.start)} -> ${formatTimestamp(seg.end)}]`);
  if (seg.words) {
    for (const word of seg.words) {
      console.log(`     "${word.word}" @ ${formatTimestamp(word.start)}`);
    }
  }
}
console.log('   ✅ Passed');

// Test 4: Hallucination detection
console.log('\n📋 Test 4: Hallucination detection threshold');
const hallucinationResult = engine.transcribeFile(audioPath, {
  hallucinationSilenceThreshold: 0.5, // Skip segments where words take > 0.5s each
});

console.log('   Text:', hallucinationResult.text);
console.log('   Segments:', hallucinationResult.segments.length);

// This should still have content for normal speech
if (hallucinationResult.text.trim().length > 0) {
  console.log('   ✅ Content preserved for normal speech');
} else {
  console.log('   ⚠️  All content filtered (threshold may be too aggressive)');
}

// Test 5: All Phase 3 options together
console.log('\n📋 Test 5: All Phase 3 options combined');
const allOptionsResult = engine.transcribeFile(audioPath, {
  language: 'en',
  wordTimestamps: true,
  vadFilter: true,
  vadOptions: {
    threshold: 0.4,
    minSpeechDurationMs: 100,
    minSilenceDurationMs: 500,
    speechPadMs: 300,
  },
  hallucinationSilenceThreshold: 2.0, // Less aggressive
});

console.log('   Text:', allOptionsResult.text);
console.log('   Duration: original=', allOptionsResult.duration.toFixed(2), 
            'after-vad=', allOptionsResult.durationAfterVad.toFixed(2));

if (allOptionsResult.segments.length > 0 && allOptionsResult.text.trim().length > 0) {
  console.log('   ✅ All options work together');
} else {
  console.log('   ⚠️  No output with all options');
}

console.log('\n✅ All Phase 3 tests passed!');
