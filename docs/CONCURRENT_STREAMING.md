# Concurrent Streaming Transcription

This document describes how to achieve concurrent streaming transcription with `faster-whisper-rs`.

## Overview

The library provides two approaches for handling multiple concurrent streams:

1. **WorkerPoolBatcher** - Uses worker threads for true parallelism
2. **StreamingBatcher** - Single-engine with request queuing

## Performance Characteristics

### Single-Threaded Performance (Tiny Model, CPU)

| Audio Duration | Processing Time | RTF |
|---------------|-----------------|-----|
| 0.5s | ~287ms | 0.57 |
| 1.0s | ~324ms | 0.32 |
| 2.0s | ~315ms | 0.16 |
| 5.0s | ~432ms | 0.09 |

*RTF = Real-Time Factor (processing time / audio duration). RTF < 1.0 means faster than real-time.*

### Multi-Worker Performance

When running multiple workers in parallel, CPU contention affects performance:

| Workers | Throughput (x real-time) | Notes |
|---------|-------------------------|-------|
| 1 | ~6x | Single engine, optimal CPU usage |
| 2 | ~9x | Best on most systems |
| 4 | ~8x | Diminishing returns start |
| 8 | ~5x | Significant contention |
| 10 (1 thread each) | ~10x | Limited threads reduces contention |

*Tested on 10-core Apple Silicon CPU with tiny model*

## Recommendations

### For Maximum Throughput

Use `numWorkers` equal to your CPU cores, with `cpuThreadsPerWorker: 1`:

```javascript
const { createWorkerPool } = require('faster-whisper-rs');

const pool = createWorkerPool('./models/tiny', {
  numWorkers: 10,          // Equal to CPU cores
  cpuThreadsPerWorker: 1,  // Limit internal threads
  language: 'en',
  beamSize: 5,
});

await pool.init();

// Process chunks from multiple streams
const result = await pool.transcribeChunk(streamId, audioSamples, startTime);
```

### For Real-Time Streaming

The number of concurrent real-time streams you can handle depends on:

1. **CPU cores** - More cores = more parallel workers
2. **Model size** - Smaller models are faster
3. **Chunk duration** - Longer chunks are more efficient

Example capacity (10-core CPU, tiny model, 2-second chunks):
- **6 concurrent streams**: Latency stays under 1 second
- **10 concurrent streams**: Some latency (~3 seconds)
- **100 concurrent streams**: Requires 10x more compute (GPU or multi-machine)

### For 100+ Concurrent Streams

To handle 100 concurrent real-time streams, you need approximately:

**Option 1: Multiple Machines**
- ~17 machines like the test system
- Use a load balancer to distribute streams

**Option 2: GPU**
- NVIDIA GPU with CUDA support
- Set `device: 'cuda'` in model options
- Expected 10-50x speedup depending on GPU

**Option 3: Longer Chunks**
- Instead of 2-second chunks every 2 seconds
- Use 10-second chunks every 10 seconds
- Reduces overhead and increases efficiency

## API Reference

### WorkerPoolBatcher

```javascript
const { createWorkerPool } = require('faster-whisper-rs');

const pool = createWorkerPool(modelPath, {
  numWorkers: 8,           // Number of worker threads
  cpuThreadsPerWorker: 1,  // CPU threads per worker (0 = auto)
  language: 'en',          // Language code
  beamSize: 5,             // Beam size for decoding
  wordTimestamps: false,   // Include word-level timestamps
});

// Initialize (loads model in each worker)
await pool.init();

// Create a stream
const streamId = pool.createStream();

// Transcribe a chunk
const result = await pool.transcribeChunk(streamId, audioSamples, audioStartTime);
// Returns: { text, segments, processingTimeMs, queueTimeMs, ... }

// Close stream when done
pool.closeStream(streamId);

// Get stats
const stats = pool.getStats();
// Returns: { totalChunks, totalProcessingMs, pendingRequests, ... }

// Cleanup
pool.destroy();
```

### StreamingBatcher (Single Engine)

```javascript
const { createBatcher } = require('faster-whisper-rs');

const batcher = createBatcher(modelPath, {
  maxBatchSize: 8,    // Max chunks to process together
  maxWaitMs: 50,      // Max wait before processing partial batch
  language: 'en',
  beamSize: 5,
});

// Create stream and transcribe (same API as WorkerPoolBatcher)
const streamId = batcher.createStream();
const result = await batcher.transcribeChunk(streamId, samples, startTime);
```

## Testing Concurrent Streaming

Run the provided stress tests:

```bash
# Throughput benchmark
node stress-tests/throughput-bench.js --chunks=100 --workers=8

# Thread configuration optimization
node stress-tests/thread-config-bench.js

# Realtime streaming simulation
node stress-tests/realtime-streaming.test.js --streams=6 --duration=10 --workers=6
```

## Limitations

1. **CPU Contention**: Multiple workers compete for CPU resources
2. **Memory**: Each worker loads the model separately (~150MB for tiny)
3. **No True Batching**: Current implementation doesn't batch across streams at the model level (would require low-level API access)

## Future Improvements

1. **True Batched Inference**: Batch mel spectrograms from multiple streams for single forward pass
2. **GPU Support**: CUDA acceleration for much higher throughput
3. **Dynamic Scaling**: Automatically adjust workers based on load
