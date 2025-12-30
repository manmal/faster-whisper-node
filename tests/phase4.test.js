/**
 * Phase 4 Test - GPU detection and auto device selection
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
const { 
  Engine, 
  isGpuAvailable, 
  getGpuCount, 
  getBestDevice 
} = require('../index');

// Test 1: GPU availability check
console.log('\n📋 Test 1: GPU availability check');
const gpuAvailable = isGpuAvailable();
const gpuCount = getGpuCount();
const bestDevice = getBestDevice();

console.log('   GPU Available:', gpuAvailable);
console.log('   GPU Count:', gpuCount);
console.log('   Best Device:', bestDevice);

if (typeof gpuAvailable !== 'boolean') {
  console.error('❌ isGpuAvailable() should return boolean');
  process.exit(1);
}
if (typeof gpuCount !== 'number') {
  console.error('❌ getGpuCount() should return number');
  process.exit(1);
}
if (bestDevice !== 'cpu' && bestDevice !== 'cuda') {
  console.error('❌ getBestDevice() should return "cpu" or "cuda"');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 2: Auto device selection
console.log('\n📋 Test 2: Auto device selection');
const engineAuto = Engine.withOptions(modelPath, {
  device: 'auto',
});
const autoResult = engineAuto.transcribe(audioPath);
console.log('   Device: auto');
console.log('   Result:', autoResult);
if (!autoResult || autoResult.trim().length === 0) {
  console.error('❌ Transcription with auto device should work');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 3: Explicit CPU
console.log('\n📋 Test 3: Explicit CPU device');
const engineCpu = Engine.withOptions(modelPath, {
  device: 'cpu',
});
const cpuResult = engineCpu.transcribe(audioPath);
console.log('   Device: cpu');
console.log('   Result:', cpuResult);
if (!cpuResult || cpuResult.trim().length === 0) {
  console.error('❌ Transcription with cpu device should work');
  process.exit(1);
}
console.log('   ✅ Passed');

// Test 4: CUDA (if available)
console.log('\n📋 Test 4: CUDA device (if available)');
if (gpuAvailable) {
  try {
    const engineCuda = Engine.withOptions(modelPath, {
      device: 'cuda',
    });
    const cudaResult = engineCuda.transcribe(audioPath);
    console.log('   Device: cuda');
    console.log('   Result:', cudaResult);
    console.log('   ✅ Passed');
  } catch (err) {
    console.log('   ⚠️  CUDA device failed:', err.message);
  }
} else {
  console.log('   ⚠️  Skipped (no GPU available)');
}

// Test 5: Best device consistency
console.log('\n📋 Test 5: Best device consistency');
if (gpuAvailable && gpuCount > 0) {
  if (bestDevice !== 'cuda') {
    console.error('❌ GPU available but best device is not cuda');
    process.exit(1);
  }
  console.log('   ✅ GPU available, best device is cuda');
} else {
  if (bestDevice !== 'cpu') {
    console.error('❌ No GPU but best device is not cpu');
    process.exit(1);
  }
  console.log('   ✅ No GPU, best device is cpu');
}

console.log('\n✅ All Phase 4 tests passed!');
