/**
 * WorkerPoolBatcher - True parallel concurrent transcription
 * 
 * Uses worker_threads to achieve real parallelism. Each worker has its own
 * Engine instance. Work is distributed round-robin to workers for maximum
 * throughput.
 */

const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { EventEmitter } = require('events');
const path = require('path');
const os = require('os');

// ============== Worker Code ==============
if (!isMainThread) {
  const { Engine } = require(workerData.enginePath);
  
  // Create engine in worker with thread limit
  const engine = Engine.withOptions(workerData.modelPath, {
    cpuThreads: workerData.cpuThreads || 0,  // 0 = auto
  });
  
  parentPort.on('message', (msg) => {
    if (msg.type === 'transcribe') {
      const { requestId, samples, options } = msg;
      const startTime = Date.now();
      
      try {
        const result = engine.transcribeSamples(samples, options);
        const processingTime = Date.now() - startTime;
        
        parentPort.postMessage({
          type: 'result',
          requestId,
          result: {
            text: result.text,
            segments: result.segments,
          },
          processingTimeMs: processingTime,
        });
      } catch (error) {
        parentPort.postMessage({
          type: 'error',
          requestId,
          error: error.message,
        });
      }
    } else if (msg.type === 'exit') {
      process.exit(0);
    }
  });
  
  parentPort.postMessage({ type: 'ready' });
}

// ============== Main Thread Code ==============

class WorkerPoolBatcher extends EventEmitter {
  /**
   * Create a WorkerPoolBatcher
   * @param {string} modelPath - Path to the model
   * @param {Object} options - Options
   * @param {number} [options.numWorkers] - Number of worker threads (default: CPU cores)
   * @param {number} [options.cpuThreadsPerWorker] - CPU threads per worker (default: auto)
   * @param {string} [options.language='en'] - Language for transcription
   * @param {number} [options.beamSize=5] - Beam size
   * @param {boolean} [options.wordTimestamps=false] - Include word timestamps
   */
  constructor(modelPath, options = {}) {
    super();
    
    if (!isMainThread) {
      throw new Error('WorkerPoolBatcher must be created from main thread');
    }
    
    this.modelPath = modelPath;
    this.options = {
      numWorkers: options.numWorkers || Math.min(os.cpus().length, 8),
      cpuThreadsPerWorker: options.cpuThreadsPerWorker || 0,
      language: options.language || 'en',
      beamSize: options.beamSize || 5,
      wordTimestamps: options.wordTimestamps || false,
    };
    
    /** @type {Worker[]} */
    this.workers = [];
    
    /** @type {Map<number, {resolve, reject, submitTime}>} */
    this.pendingRequests = new Map();
    
    /** @type {number} */
    this.nextRequestId = 0;
    
    /** @type {number} */
    this.nextWorkerIdx = 0;
    
    /** @type {boolean} */
    this.initialized = false;
    
    /** @type {Promise<void>} */
    this._initPromise = null;
    
    /** @type {number} */
    this.nextStreamId = 0;
    
    /** @type {Map<number, Object>} */
    this.streams = new Map();
    
    // Stats
    this.stats = {
      totalChunks: 0,
      totalProcessingMs: 0,
      workersReady: 0,
    };
  }
  
  /**
   * Initialize the worker pool
   * @returns {Promise<void>}
   */
  async init() {
    if (this.initialized) return;
    if (this._initPromise) return this._initPromise;
    
    this._initPromise = this._initWorkers();
    await this._initPromise;
    this.initialized = true;
  }
  
  async _initWorkers() {
    const enginePath = path.resolve(__dirname, 'index.js');
    const modelPathAbs = path.resolve(this.modelPath);
    
    const workerPromises = [];
    
    for (let i = 0; i < this.options.numWorkers; i++) {
      const workerPromise = new Promise((resolve, reject) => {
        const worker = new Worker(__filename, {
          workerData: {
            enginePath,
            modelPath: modelPathAbs,
            cpuThreads: this.options.cpuThreadsPerWorker,
          },
        });
        
        const onMessage = (msg) => {
          if (msg.type === 'ready') {
            worker.off('message', onMessage);
            worker.on('message', this._handleWorkerMessage.bind(this));
            this.stats.workersReady++;
            resolve(worker);
          }
        };
        
        worker.on('message', onMessage);
        worker.on('error', (err) => {
          console.error(`Worker ${i} error:`, err);
          reject(err);
        });
        
        this.workers.push(worker);
      });
      
      workerPromises.push(workerPromise);
    }
    
    await Promise.all(workerPromises);
    this.emit('ready', { numWorkers: this.workers.length });
  }
  
  _handleWorkerMessage(msg) {
    if (msg.type === 'result' || msg.type === 'error') {
      const pending = this.pendingRequests.get(msg.requestId);
      if (pending) {
        this.pendingRequests.delete(msg.requestId);
        
        if (msg.type === 'error') {
          pending.reject(new Error(msg.error));
        } else {
          this.stats.totalChunks++;
          this.stats.totalProcessingMs += msg.processingTimeMs;
          
          pending.resolve({
            ...msg.result,
            processingTimeMs: msg.processingTimeMs,
            queueTimeMs: Date.now() - pending.submitTime - msg.processingTimeMs,
          });
        }
      }
    }
  }
  
  /**
   * Create a new stream
   * @returns {number} Stream ID
   */
  createStream() {
    const streamId = this.nextStreamId++;
    this.streams.set(streamId, { id: streamId, chunkCount: 0 });
    return streamId;
  }
  
  /**
   * Close a stream
   * @param {number} streamId
   */
  closeStream(streamId) {
    this.streams.delete(streamId);
  }
  
  /**
   * Transcribe audio chunk
   * @param {number} streamId - Stream ID
   * @param {Float32Array|number[]} samples - Audio samples
   * @param {number} [audioStartS=0] - Start time in audio
   * @returns {Promise<Object>} Transcription result
   */
  async transcribeChunk(streamId, samples, audioStartS = 0) {
    if (!this.initialized) {
      await this.init();
    }
    
    const stream = this.streams.get(streamId);
    if (!stream) {
      throw new Error(`Stream ${streamId} not found`);
    }
    
    const chunkId = stream.chunkCount++;
    const samplesArray = Array.from(samples);
    const audioDurationS = samplesArray.length / 16000;
    
    return new Promise((resolve, reject) => {
      const requestId = this.nextRequestId++;
      const submitTime = Date.now();
      
      this.pendingRequests.set(requestId, { resolve, reject, submitTime });
      
      // Round-robin to workers
      const workerIdx = this.nextWorkerIdx;
      this.nextWorkerIdx = (this.nextWorkerIdx + 1) % this.workers.length;
      
      this.workers[workerIdx].postMessage({
        type: 'transcribe',
        requestId,
        samples: samplesArray,
        options: {
          language: this.options.language,
          beamSize: this.options.beamSize,
          wordTimestamps: this.options.wordTimestamps,
        },
      });
    }).then(result => ({
      streamId,
      chunkId,
      audioStartS,
      audioDurationS,
      ...result,
    }));
  }
  
  /**
   * Get statistics
   * @returns {Object}
   */
  getStats() {
    return {
      ...this.stats,
      pendingRequests: this.pendingRequests.size,
      activeStreams: this.streams.size,
      avgProcessingMs: this.stats.totalChunks > 0
        ? this.stats.totalProcessingMs / this.stats.totalChunks
        : 0,
    };
  }
  
  /**
   * Destroy the worker pool
   */
  destroy() {
    for (const worker of this.workers) {
      worker.postMessage({ type: 'exit' });
    }
    this.workers = [];
    
    // Reject pending requests
    for (const [, pending] of this.pendingRequests) {
      pending.reject(new Error('Worker pool destroyed'));
    }
    this.pendingRequests.clear();
    this.streams.clear();
  }
}

/**
 * Create a WorkerPoolBatcher
 * @param {string} modelPath
 * @param {Object} [options]
 * @returns {WorkerPoolBatcher}
 */
function createWorkerPool(modelPath, options = {}) {
  return new WorkerPoolBatcher(modelPath, options);
}

module.exports = {
  WorkerPoolBatcher,
  createWorkerPool,
};
