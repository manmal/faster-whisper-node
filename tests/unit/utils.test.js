/**
 * Unit tests for utility functions
 */
const { describe, it } = require('node:test');
const assert = require('node:assert');

const { 
  formatTimestamp,
  availableModels,
  getCacheDir,
  getModelPath,
  isGpuAvailable,
  getGpuCount,
  getBestDevice
} = require('../../index.js');

describe('formatTimestamp', () => {
  it('should format seconds to MM:SS.mmm', () => {
    assert.strictEqual(formatTimestamp(0), '00:00.000');
    assert.strictEqual(formatTimestamp(5.5), '00:05.500');
    assert.strictEqual(formatTimestamp(65.5), '01:05.500');
    assert.strictEqual(formatTimestamp(125.123), '02:05.123');
  });

  it('should format to HH:MM:SS.mmm when hours present', () => {
    assert.strictEqual(formatTimestamp(3600), '01:00:00.000');
    assert.strictEqual(formatTimestamp(3661.5), '01:01:01.500');
    assert.strictEqual(formatTimestamp(7325.999), '02:02:05.999');
  });

  it('should format to HH:MM:SS.mmm when alwaysIncludeHours=true', () => {
    assert.strictEqual(formatTimestamp(0, true), '00:00:00.000');
    assert.strictEqual(formatTimestamp(65.5, true), '00:01:05.500');
    assert.strictEqual(formatTimestamp(125.123, true), '00:02:05.123');
  });

  it('should handle edge cases', () => {
    assert.strictEqual(formatTimestamp(0.001), '00:00.001');
    assert.strictEqual(formatTimestamp(0.999), '00:00.999');
    assert.strictEqual(formatTimestamp(59.999), '00:59.999');
    assert.strictEqual(formatTimestamp(60), '01:00.000');
  });
});

describe('availableModels', () => {
  it('should return array of model names', () => {
    const models = availableModels();
    assert.ok(Array.isArray(models), 'Should return array');
    assert.ok(models.length > 0, 'Should have at least one model');
  });

  it('should include common model sizes', () => {
    const models = availableModels();
    assert.ok(models.includes('tiny'), 'Should include tiny');
    assert.ok(models.includes('tiny.en'), 'Should include tiny.en');
    assert.ok(models.includes('base'), 'Should include base');
    assert.ok(models.includes('small'), 'Should include small');
    assert.ok(models.includes('medium'), 'Should include medium');
    assert.ok(models.includes('large-v2'), 'Should include large-v2');
    assert.ok(models.includes('large-v3'), 'Should include large-v3');
  });

  it('should include English-only models', () => {
    const models = availableModels();
    const englishModels = models.filter(m => m.endsWith('.en'));
    assert.ok(englishModels.length >= 4, 'Should have at least 4 English-only models');
  });
});

describe('getCacheDir', () => {
  it('should return a string path', () => {
    const dir = getCacheDir();
    assert.ok(typeof dir === 'string', 'Should return string');
    assert.ok(dir.length > 0, 'Should not be empty');
  });

  it('should include package name in path', () => {
    const dir = getCacheDir();
    assert.ok(dir.includes('faster-whisper-node'), 'Should include package name');
  });
});

describe('getModelPath', () => {
  it('should return path for valid model size', () => {
    const path = getModelPath('tiny');
    assert.ok(typeof path === 'string', 'Should return string');
    assert.ok(path.includes('tiny'), 'Should include model name');
  });

  it('should return paths for all available models', () => {
    const models = availableModels();
    for (const model of models) {
      const modelPath = getModelPath(model);
      assert.ok(modelPath.includes(model), `Path for ${model} should include model name`);
    }
  });
});

describe('GPU detection', () => {
  describe('isGpuAvailable', () => {
    it('should return boolean', () => {
      const result = isGpuAvailable();
      assert.ok(typeof result === 'boolean', 'Should return boolean');
    });
  });

  describe('getGpuCount', () => {
    it('should return non-negative number', () => {
      const count = getGpuCount();
      assert.ok(typeof count === 'number', 'Should return number');
      assert.ok(count >= 0, 'Should be non-negative');
    });

    it('should be consistent with isGpuAvailable', () => {
      const available = isGpuAvailable();
      const count = getGpuCount();
      
      if (available) {
        assert.ok(count > 0, 'If GPU available, count should be > 0');
      } else {
        assert.strictEqual(count, 0, 'If no GPU, count should be 0');
      }
    });
  });

  describe('getBestDevice', () => {
    it('should return "cpu" or "cuda"', () => {
      const device = getBestDevice();
      assert.ok(['cpu', 'cuda'].includes(device), 'Should be "cpu" or "cuda"');
    });

    it('should be consistent with isGpuAvailable', () => {
      const available = isGpuAvailable();
      const device = getBestDevice();
      
      if (available) {
        assert.strictEqual(device, 'cuda', 'If GPU available, best device should be cuda');
      } else {
        assert.strictEqual(device, 'cpu', 'If no GPU, best device should be cpu');
      }
    });
  });
});
