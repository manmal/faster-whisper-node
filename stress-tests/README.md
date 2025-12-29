# Stress Tests

Comprehensive stress testing suite for faster-whisper-node.

## Available Tests

| Test | Description | Command |
|------|-------------|---------|
| `concurrent.test.js` | HTTP server concurrent requests | `npm run stress:test` |
| `worker-pool.test.js` | Worker threads true parallelism | `npm run stress:workers` |
| `multi-engine.test.js` | Multiple engines in same process | `npm run stress:engines` |
| `memory.test.js` | Memory stability over many requests | `npm run stress:memory` |
| `feature-stress.test.js` | All API features under load | `npm run stress:features` |

## Quick Start

```bash
# Run all stress tests
npm run stress:all

# Or run individual tests:

# 1. HTTP Server stress test (requires server running)
npm run stress:server &   # Start server
npm run stress:test       # Run test

# 2. Worker pool (true parallelism)
npm run stress:workers

# 3. Multi-engine (same process)
npm run stress:engines

# 4. Memory stability
npm run stress:memory

# 5. API feature stress
npm run stress:features
```

## HTTP Server Tests

### Server (`server.js`)

The sample server provides:
- Engine pool for concurrent requests (default: 4 engines)
- Request queuing when all engines are busy
- Statistics endpoint

#### Configuration

Environment variables:
- `PORT` - Server port (default: 3000)
- `MODEL_PATH` - Path to model (default: ./models/tiny)
- `POOL_SIZE` - Number of engine instances (default: 4)

```bash
# Run with 8 engine instances on port 8080
POOL_SIZE=8 PORT=8080 node stress-tests/server.js
```

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Server statistics |
| `/transcribe` | POST | Transcribe audio |

### Client (`concurrent.test.js`)

```bash
node stress-tests/concurrent.test.js [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--requests=N` | 100 | Concurrent requests per batch |
| `--batches=N` | 1 | Number of batches |
| `--server=URL` | http://localhost:3000 | Server URL |
| `--audio=PATH` | ./tests/fixtures/hello.wav | Audio file |

## Worker Pool Test

Tests true parallelism using Node.js worker threads, where each worker has its own Engine instance.

```bash
node stress-tests/worker-pool.test.js [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--requests=N` | 100 | Total requests |
| `--workers=N` | CPU count | Worker thread count |
| `--audio=PATH` | ./tests/fixtures/hello.wav | Audio file |

### Example Output

```
============================================================
Worker Pool Stress Test
============================================================
Total requests:  100
Worker threads:  8
Model:           ./models/tiny
Audio:           ./tests/fixtures/hello.wav

Initializing workers...
  Worker 1/8 ready
  ...
  Worker 8/8 ready

Starting 100 requests across 8 workers...

Progress: 100/100 (100 ok)

============================================================
Results
============================================================
Total requests:      100
Successful:          100 (100.0%)
Failed:              0
Total time:          6234 ms
Throughput:          16.04 req/s

Latency:
  Min:               412 ms
  Max:               623 ms
  Mean:              523 ms
  Median (p50):      518 ms
  p90:               589 ms
  p99:               618 ms

✅ Worker pool test complete!
```

## Multi-Engine Test

Tests running multiple Engine instances in the same process to verify thread safety.

```bash
node stress-tests/multi-engine.test.js [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--engines=N` | 4 | Number of engine instances |
| `--requests=N` | 100 | Total requests |
| `--sequential` | false | Run sequentially vs interleaved |

## Memory Test

Monitors memory usage across many transcriptions to detect potential leaks.

```bash
node stress-tests/memory.test.js [options]

# With garbage collection visibility
node --expose-gc stress-tests/memory.test.js --gc
```

| Option | Default | Description |
|--------|---------|-------------|
| `--iterations=N` | 100 | Number of transcriptions |
| `--gc` | false | Force GC between batches |
| `--verbose` | false | Show detailed progress |

### Example Output

```
============================================================
Memory Stability Stress Test
============================================================
Iterations:      100
Model:           ./models/tiny
Audio:           ./tests/fixtures/hello.wav
Force GC:        yes (every 10 iterations)

Loading engine...
Initial memory: 4.23 MB heap, 145.67 MB RSS

Running 100 transcriptions...

Progress: 100/100 - heap: 5.12 MB (+0.89 MB)

============================================================
Results
============================================================
Iterations:          100
Total time:          52341 ms
Throughput:          1.91 transcriptions/s

Memory usage:
  Initial heap:      4.23 MB
  Final heap:        5.12 MB
  Heap growth:       0.89 MB (21.0%)

Memory trend analysis:
  Avg heap growth/10 iters: 89.12 KB

✅ Memory usage appears stable

Memory timeline (heap used):
     0: ████████████████████████████████████████ 4.23 MB
    10: ████████████████████████████████████████░ 4.45 MB
    ...
   100: █████████████████████████████████████████ 5.12 MB

✅ Memory test complete!
```

## Feature Stress Test

Exercises all API features repeatedly to ensure stability.

```bash
node stress-tests/feature-stress.test.js [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--iterations=N` | 20 | Iterations per feature |
| `--verbose` | false | Show detailed output |

Tests include:
- Utility functions (formatTimestamp, availableModels, etc.)
- Audio decoding (decodeAudio, decodeAudioBuffer)
- Engine properties (samplingRate, isMultilingual, numLanguages)
- All transcription methods (transcribe, transcribeFile, transcribeBuffer, etc.)
- Word timestamps
- Language detection
- VAD (Voice Activity Detection)
- Result structure validation

## Performance Tuning

### Engine Pool Size

Tune based on:
- Available CPU cores
- Memory (each engine ~100-300MB depending on model)
- Expected concurrency

```bash
# More engines = better throughput, more memory
POOL_SIZE=8 node stress-tests/server.js
```

### Model Size

| Model | Transcription Time* | Memory |
|-------|---------------------|--------|
| tiny | ~500ms | ~75MB |
| base | ~800ms | ~150MB |
| small | ~1500ms | ~500MB |

*For ~1s audio on Apple M1

### Best Practices

1. **Match pool size to concurrency** - 4 engines can handle ~25 queued requests each efficiently.

2. **Use worker threads for true parallelism** - Multiple engines in one thread still share the CPU.

3. **Specify language** - Using `language: 'en'` skips detection, ~2x faster.

4. **Monitor memory** - Use `memory.test.js` to check for leaks before production.

5. **Use VAD for silence** - `vadFilter: true` skips silent portions, faster for sparse audio.
