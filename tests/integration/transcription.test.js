/**
 * Integration tests for transcription features
 */
const { describe, it, before } = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');

const { Engine, formatTimestamp, decodeAudio } = require('../../index.js');

const modelPath = path.join(__dirname, '../../models/tiny');
const hasModel = fs.existsSync(path.join(modelPath, 'model.bin'));
const fixturesDir = path.join(__dirname, '../fixtures');

describe('Transcription Integration', { skip: !hasModel && 'Model not available' }, () => {
  let engine;
  
  before(() => {
    engine = new Engine(modelPath);
  });

  describe('Basic transcription', () => {
    it('should transcribe WAV file (simple API)', () => {
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      
      assert.ok(typeof result === 'string', 'Should return string');
      assert.ok(result.trim().length > 0, 'Should have content');
    });

    it('should transcribe WAV file (full API)', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'));
      
      assert.ok(result.segments, 'Should have segments');
      assert.ok(Array.isArray(result.segments), 'Segments should be array');
      assert.ok(result.text, 'Should have text');
      assert.ok(typeof result.duration === 'number', 'Should have duration');
      assert.ok(result.duration > 0, 'Duration should be positive');
    });

    it('should transcribe MP3 file', () => {
      const result = engine.transcribe(path.join(fixturesDir, 'hello.mp3'));
      
      assert.ok(typeof result === 'string', 'Should return string');
      assert.ok(result.trim().length > 0, 'Should have content');
    });

    it('should transcribe FLAC file', () => {
      const result = engine.transcribe(path.join(fixturesDir, 'hello.flac'));
      
      assert.ok(typeof result === 'string', 'Should return string');
      assert.ok(result.trim().length > 0, 'Should have content');
    });

    it('should transcribe OGG file', () => {
      const result = engine.transcribe(path.join(fixturesDir, 'hello.ogg'));
      
      assert.ok(typeof result === 'string', 'Should return string');
      assert.ok(result.trim().length > 0, 'Should have content');
    });
  });

  describe('Segment structure', () => {
    it('should have correct segment properties', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'));
      
      assert.ok(result.segments.length > 0, 'Should have at least one segment');
      
      const segment = result.segments[0];
      assert.ok(typeof segment.id === 'number', 'Should have id');
      assert.ok(typeof segment.start === 'number', 'Should have start');
      assert.ok(typeof segment.end === 'number', 'Should have end');
      assert.ok(typeof segment.text === 'string', 'Should have text');
      assert.ok(Array.isArray(segment.tokens), 'Should have tokens');
      assert.ok(typeof segment.temperature === 'number', 'Should have temperature');
      assert.ok(typeof segment.avgLogprob === 'number', 'Should have avgLogprob');
      assert.ok(typeof segment.compressionRatio === 'number', 'Should have compressionRatio');
      assert.ok(typeof segment.noSpeechProb === 'number', 'Should have noSpeechProb');
    });

    it('should have valid segment timing', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'));
      
      for (const segment of result.segments) {
        assert.ok(segment.start >= 0, 'Start should be >= 0');
        assert.ok(segment.end > segment.start, 'End should be > start');
        assert.ok(segment.end <= result.duration + 0.1, 'End should be <= duration');
      }
    });

    it('should have consistent text between text and segments', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'));
      
      const segmentText = result.segments.map(s => s.text).join('').trim();
      assert.strictEqual(result.text.trim(), segmentText, 'Text should match joined segments');
    });
  });

  describe('Transcription options', () => {
    it('should accept language option', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        language: 'en'
      });
      
      assert.ok(result.text, 'Should have text');
    });

    it('should accept beamSize option', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        beamSize: 3
      });
      
      assert.ok(result.text, 'Should have text');
    });

    it('should accept temperature option', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        temperature: 0.0
      });
      
      assert.ok(result.text, 'Should have text');
    });

    it('should accept multiple options together', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        language: 'en',
        beamSize: 5,
        temperature: 0.0,
        suppressBlank: true
      });
      
      assert.ok(result.text, 'Should have text');
    });
  });

  describe('Input methods', () => {
    it('should transcribe from Buffer', () => {
      const buffer = fs.readFileSync(path.join(fixturesDir, 'hello.wav'));
      const result = engine.transcribeBuffer(buffer);
      
      assert.ok(result.text, 'Should have text');
      assert.ok(result.segments.length > 0, 'Should have segments');
    });

    it('should transcribe from samples', () => {
      const samples = decodeAudio(path.join(fixturesDir, 'hello.wav'));
      const result = engine.transcribeSamples(samples);
      
      assert.ok(result.text, 'Should have text');
      assert.ok(result.segments.length > 0, 'Should have segments');
    });

    it('should produce consistent results across input methods', () => {
      const audioPath = path.join(fixturesDir, 'hello.wav');
      
      const fileResult = engine.transcribeFile(audioPath);
      
      const buffer = fs.readFileSync(audioPath);
      const bufferResult = engine.transcribeBuffer(buffer);
      
      const samples = decodeAudio(audioPath);
      const samplesResult = engine.transcribeSamples(samples);
      
      // All should produce text (may differ slightly due to processing)
      assert.ok(fileResult.text.trim(), 'File result should have text');
      assert.ok(bufferResult.text.trim(), 'Buffer result should have text');
      assert.ok(samplesResult.text.trim(), 'Samples result should have text');
    });
  });

  describe('Word timestamps', () => {
    it('should return word-level timestamps when enabled', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        wordTimestamps: true
      });
      
      assert.ok(result.segments.length > 0, 'Should have segments');
      
      const segment = result.segments[0];
      assert.ok(segment.words, 'Segment should have words');
      assert.ok(Array.isArray(segment.words), 'Words should be array');
    });

    it('should have correct word properties', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        wordTimestamps: true
      });
      
      const segment = result.segments[0];
      if (segment.words && segment.words.length > 0) {
        const word = segment.words[0];
        assert.ok(typeof word.word === 'string', 'Should have word text');
        assert.ok(typeof word.start === 'number', 'Should have start time');
        assert.ok(typeof word.end === 'number', 'Should have end time');
        assert.ok(typeof word.probability === 'number', 'Should have probability');
      }
    });

    it('should have valid word timing', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        wordTimestamps: true
      });
      
      for (const segment of result.segments) {
        if (segment.words) {
          for (const word of segment.words) {
            assert.ok(word.start >= 0, 'Word start should be >= 0');
            assert.ok(word.end >= word.start, 'Word end should be >= start');
            assert.ok(word.probability >= 0 && word.probability <= 1, 'Probability should be 0-1');
          }
        }
      }
    });
  });

  describe('VAD filtering', () => {
    it('should apply VAD filter when enabled', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        vadFilter: true
      });
      
      assert.ok(result.text, 'Should have text');
      assert.ok(typeof result.duration === 'number', 'Should have duration');
      assert.ok(typeof result.durationAfterVad === 'number', 'Should have durationAfterVad');
      assert.ok(result.durationAfterVad <= result.duration, 'VAD should not increase duration');
    });

    it('should accept VAD options', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        vadFilter: true,
        vadOptions: {
          threshold: 0.5,
          minSpeechDurationMs: 100,
          speechPadMs: 200
        }
      });
      
      assert.ok(result.text, 'Should have text');
    });

    it('should work with word timestamps', () => {
      const result = engine.transcribeFile(path.join(fixturesDir, 'hello.wav'), {
        vadFilter: true,
        wordTimestamps: true
      });
      
      assert.ok(result.text, 'Should have text');
      if (result.segments.length > 0) {
        assert.ok('words' in result.segments[0], 'Should have words in segment');
      }
    });
  });

  describe('Engine properties', () => {
    it('should return correct sampling rate', () => {
      const rate = engine.samplingRate();
      assert.strictEqual(rate, 16000, 'Sampling rate should be 16000');
    });

    it('should report multilingual status', () => {
      const isMultilingual = engine.isMultilingual();
      assert.ok(typeof isMultilingual === 'boolean', 'Should return boolean');
      // tiny model is multilingual
      assert.strictEqual(isMultilingual, true, 'Tiny model should be multilingual');
    });

    it('should report number of languages', () => {
      const numLanguages = engine.numLanguages();
      assert.ok(typeof numLanguages === 'number', 'Should return number');
      assert.ok(numLanguages > 0, 'Should support at least one language');
    });
  });

  describe('Language detection', () => {
    it('should detect language', () => {
      const result = engine.detectLanguage(path.join(fixturesDir, 'hello.wav'));
      
      assert.ok(typeof result.language === 'string', 'Should have language');
      assert.ok(typeof result.probability === 'number', 'Should have probability');
    });

    it('should detect language from buffer', () => {
      const buffer = fs.readFileSync(path.join(fixturesDir, 'hello.wav'));
      const result = engine.detectLanguageBuffer(buffer);
      
      assert.ok(typeof result.language === 'string', 'Should have language');
      assert.ok(typeof result.probability === 'number', 'Should have probability');
    });
  });
});
