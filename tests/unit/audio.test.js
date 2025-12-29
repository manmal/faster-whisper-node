/**
 * Unit tests for audio decoding functions
 */
const { describe, it } = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');

const { decodeAudio, decodeAudioBuffer } = require('../../index.js');

const fixturesDir = path.join(__dirname, '../fixtures');

describe('decodeAudio', () => {
  it('should decode WAV file', () => {
    const samples = decodeAudio(path.join(fixturesDir, 'hello.wav'));
    
    assert.ok(Array.isArray(samples), 'Should return array');
    assert.ok(samples.length > 0, 'Should have samples');
    // hello.wav is ~0.63 seconds at 16kHz = ~10000 samples
    assert.ok(samples.length > 5000, 'Should have reasonable number of samples');
    assert.ok(samples.length < 20000, 'Should not have too many samples');
  });

  it('should decode MP3 file', () => {
    const samples = decodeAudio(path.join(fixturesDir, 'hello.mp3'));
    
    assert.ok(Array.isArray(samples), 'Should return array');
    assert.ok(samples.length > 0, 'Should have samples');
  });

  it('should decode FLAC file', () => {
    const samples = decodeAudio(path.join(fixturesDir, 'hello.flac'));
    
    assert.ok(Array.isArray(samples), 'Should return array');
    assert.ok(samples.length > 0, 'Should have samples');
  });

  it('should decode OGG file', () => {
    const samples = decodeAudio(path.join(fixturesDir, 'hello.ogg'));
    
    assert.ok(Array.isArray(samples), 'Should return array');
    assert.ok(samples.length > 0, 'Should have samples');
  });

  it('should return normalized samples [-1, 1]', () => {
    const samples = decodeAudio(path.join(fixturesDir, 'hello.wav'));
    
    for (let i = 0; i < samples.length; i++) {
      assert.ok(samples[i] >= -1.0, `Sample ${i} should be >= -1`);
      assert.ok(samples[i] <= 1.0, `Sample ${i} should be <= 1`);
    }
  });

  it('should throw on nonexistent file', () => {
    assert.throws(() => {
      decodeAudio('/nonexistent/audio.wav');
    }, /Error|not found|failed|No such file/i);
  });

  it('should produce consistent output for same file', () => {
    const samples1 = decodeAudio(path.join(fixturesDir, 'hello.wav'));
    const samples2 = decodeAudio(path.join(fixturesDir, 'hello.wav'));
    
    assert.strictEqual(samples1.length, samples2.length, 'Sample counts should match');
    for (let i = 0; i < samples1.length; i++) {
      assert.strictEqual(samples1[i], samples2[i], `Sample ${i} should match`);
    }
  });
});

describe('decodeAudioBuffer', () => {
  it('should decode WAV buffer', () => {
    const buffer = fs.readFileSync(path.join(fixturesDir, 'hello.wav'));
    const samples = decodeAudioBuffer(buffer);
    
    assert.ok(Array.isArray(samples), 'Should return array');
    assert.ok(samples.length > 0, 'Should have samples');
  });

  it('should decode MP3 buffer', () => {
    const buffer = fs.readFileSync(path.join(fixturesDir, 'hello.mp3'));
    const samples = decodeAudioBuffer(buffer);
    
    assert.ok(Array.isArray(samples), 'Should return array');
    assert.ok(samples.length > 0, 'Should have samples');
  });

  it('should produce same output as decodeAudio for WAV', () => {
    const filePath = path.join(fixturesDir, 'hello.wav');
    const filesamples = decodeAudio(filePath);
    
    const buffer = fs.readFileSync(filePath);
    const bufferSamples = decodeAudioBuffer(buffer);
    
    assert.strictEqual(filesamples.length, bufferSamples.length, 'Sample counts should match');
    for (let i = 0; i < filesamples.length; i++) {
      assert.ok(
        Math.abs(filesamples[i] - bufferSamples[i]) < 0.0001,
        `Sample ${i} should match`
      );
    }
  });

  it('should throw on invalid buffer', () => {
    assert.throws(() => {
      decodeAudioBuffer(Buffer.from('not audio data'));
    }, /Error|failed|Invalid|unsupported/i);
  });

  it('should throw on empty buffer', () => {
    assert.throws(() => {
      decodeAudioBuffer(Buffer.alloc(0));
    }, /Error|failed|Invalid|empty/i);
  });
});
