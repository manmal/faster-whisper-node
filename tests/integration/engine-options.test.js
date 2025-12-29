/**
 * Integration tests for Engine.withOptions and device selection
 */
const { describe, it, before } = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');

const { Engine, isGpuAvailable, getBestDevice } = require('../../index.js');

const modelPath = path.join(__dirname, '../../models/tiny');
const hasModel = fs.existsSync(path.join(modelPath, 'model.bin'));
const fixturesDir = path.join(__dirname, '../fixtures');

describe('Engine.withOptions', { skip: !hasModel && 'Model not available' }, () => {
  describe('Device selection', () => {
    it('should create engine with CPU device', () => {
      const engine = Engine.withOptions(modelPath, {
        device: 'cpu'
      });
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });

    it('should create engine with auto device', () => {
      const engine = Engine.withOptions(modelPath, {
        device: 'auto'
      });
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });

    it('should use CUDA if available with auto device', () => {
      const gpuAvailable = isGpuAvailable();
      const bestDevice = getBestDevice();
      
      // Just verify the device selection is consistent
      if (gpuAvailable) {
        assert.strictEqual(bestDevice, 'cuda', 'Best device should be CUDA when GPU available');
      } else {
        assert.strictEqual(bestDevice, 'cpu', 'Best device should be CPU when no GPU');
      }
    });
  });

  describe('Compute type options', () => {
    it('should accept int8 compute type', () => {
      const engine = Engine.withOptions(modelPath, {
        device: 'cpu',
        computeType: 'int8'
      });
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });

    it('should accept float32 compute type', () => {
      const engine = Engine.withOptions(modelPath, {
        device: 'cpu',
        computeType: 'float32'
      });
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });

    it('should accept default compute type', () => {
      const engine = Engine.withOptions(modelPath, {
        device: 'cpu',
        computeType: 'default'
      });
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });
  });

  describe('Thread options', () => {
    it('should accept cpuThreads option', () => {
      const engine = Engine.withOptions(modelPath, {
        device: 'cpu',
        cpuThreads: 2
      });
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });

    it('should accept cpuThreads=0 for auto', () => {
      const engine = Engine.withOptions(modelPath, {
        device: 'cpu',
        cpuThreads: 0
      });
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });
  });

  describe('Multiple options combined', () => {
    it('should accept all options together', () => {
      const engine = Engine.withOptions(modelPath, {
        device: 'cpu',
        computeType: 'int8',
        cpuThreads: 4
      });
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });
  });

  describe('Empty/null options', () => {
    it('should work with undefined options', () => {
      const engine = Engine.withOptions(modelPath, undefined);
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });

    it('should work with null options', () => {
      const engine = Engine.withOptions(modelPath, null);
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });

    it('should work with empty options object', () => {
      const engine = Engine.withOptions(modelPath, {});
      
      const result = engine.transcribe(path.join(fixturesDir, 'hello.wav'));
      assert.ok(result.trim().length > 0, 'Should transcribe successfully');
    });
  });
});

describe('Multiple Engine instances', { skip: !hasModel && 'Model not available' }, () => {
  it('should support multiple simultaneous engines', () => {
    const engine1 = new Engine(modelPath);
    const engine2 = new Engine(modelPath);
    
    const result1 = engine1.transcribe(path.join(fixturesDir, 'hello.wav'));
    const result2 = engine2.transcribe(path.join(fixturesDir, 'hello.wav'));
    
    assert.ok(result1.trim().length > 0, 'Engine 1 should transcribe');
    assert.ok(result2.trim().length > 0, 'Engine 2 should transcribe');
    assert.strictEqual(result1, result2, 'Both engines should produce same result');
  });

  it('should support engines with different options', () => {
    const engine1 = Engine.withOptions(modelPath, { cpuThreads: 1 });
    const engine2 = Engine.withOptions(modelPath, { cpuThreads: 4 });
    
    const result1 = engine1.transcribe(path.join(fixturesDir, 'hello.wav'));
    const result2 = engine2.transcribe(path.join(fixturesDir, 'hello.wav'));
    
    assert.ok(result1.trim().length > 0, 'Engine 1 should transcribe');
    assert.ok(result2.trim().length > 0, 'Engine 2 should transcribe');
  });
});
