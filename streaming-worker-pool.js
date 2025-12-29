/**
 * StreamingWorkerPool - True parallel streaming transcription
 * 
 * Combines WorkerPoolBatcher parallelism with StreamingEngine's LocalAgreement algorithm.
 * Each worker runs a StreamingEngine with its own sessions.
 */

const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { EventEmitter } = require('events');
const path = require('path');
const os = require('os');

// ============== Worker Code ==============
if (!isMainThread) {
  const { StreamingEngine } = require(workerData.enginePath);
  
  // Create streaming engine in worker
  const engine = new StreamingEngine(workerData.modelPath);
  
  // Session mapping (local session ID -> engine session ID)
  const sessions = new Map();
  
  parentPort.on('message', (msg) => {
    switch (msg.type) {
      case 'createSession': {
        const engineSessionId = engine.createSession(msg.options);
        sessions.set(msg.sessionId, engineSessionId);
        parentPort.postMessage({ 
          type: 'sessionCreated', 
          requestId: msg.requestId,
          sessionId: msg.sessionId 
        });
        break;
      }
      
      case 'processAudio': {
        const engineSessionId = sessions.get(msg.sessionId);
        if (engineSessionId === undefined) {
          parentPort.postMessage({
            type: 'error',
            requestId: msg.requestId,
            error: `Session ${msg.sessionId} not found`,
          });
          break;
        }
        
        const startTime = Date.now();
        try {
          const result = engine.processAudio(engineSessionId, msg.samples);
          parentPort.postMessage({
            type: 'result',
            requestId: msg.requestId,
            sessionId: msg.sessionId,
            result,
            processingTimeMs: Date.now() - startTime,
          });
        } catch (error) {
          parentPort.postMessage({
            type: 'error',
            requestId: msg.requestId,
            error: error.message,
          });
        }
        break;
      }
      
      case 'flushSession': {
        const engineSessionId = sessions.get(msg.sessionId);
        if (engineSessionId === undefined) {
          parentPort.postMessage({
            type: 'error',
            requestId: msg.requestId,
            error: `Session ${msg.sessionId} not found`,
          });
          break;
        }
        
        try {
          const result = engine.flushSession(engineSessionId);
          parentPort.postMessage({
            type: 'flushResult',
            requestId: msg.requestId,
            sessionId: msg.sessionId,
            result,
          });
        } catch (error) {
          parentPort.postMessage({
            type: 'error',
            requestId: msg.requestId,
            error: error.message,
          });
        }
        break;
      }
      
      case 'closeSession': {
        const engineSessionId = sessions.get(msg.sessionId);
        if (engineSessionId !== undefined) {
          engine.closeSession(engineSessionId);
          sessions.delete(msg.sessionId);
        }
        parentPort.postMessage({
          type: 'sessionClosed',
          requestId: msg.requestId,
          sessionId: msg.sessionId,
        });
        break;
      }
      
      case 'exit':
        process.exit(0);
        break;
    }
  });
  
  parentPort.postMessage({ type: 'ready' });
}

// ============== Main Thread Code ==============

class StreamingWorkerPool extends EventEmitter {
  constructor(modelPath, options = {}) {
    super();
    
    if (!isMainThread) {
      throw new Error('StreamingWorkerPool must be created from main thread');
    }
    
    this.modelPath = modelPath;
    this.options = {
      numWorkers: options.numWorkers || Math.min(os.cpus().length, 8),
      ...options,
    };
    
    this.workers = [];
    this.pendingRequests = new Map();
    this.nextRequestId = 0;
    this.nextSessionId = 0;
    this.sessionWorkerMap = new Map(); // session ID -> worker index
    this.initialized = false;
    this._initPromise = null;
  }
  
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
          workerData: { enginePath, modelPath: modelPathAbs },
        });
        
        const onMessage = (msg) => {
          if (msg.type === 'ready') {
            worker.off('message', onMessage);
            worker.on('message', this._handleWorkerMessage.bind(this));
            resolve(worker);
          }
        };
        
        worker.on('message', onMessage);
        worker.on('error', reject);
        this.workers.push(worker);
      });
      
      workerPromises.push(workerPromise);
    }
    
    await Promise.all(workerPromises);
    this.emit('ready', { numWorkers: this.workers.length });
  }
  
  _handleWorkerMessage(msg) {
    const pending = this.pendingRequests.get(msg.requestId);
    if (!pending) return;
    
    this.pendingRequests.delete(msg.requestId);
    
    if (msg.type === 'error') {
      pending.reject(new Error(msg.error));
    } else {
      pending.resolve(msg);
    }
  }
  
  _sendToWorker(workerIdx, message) {
    return new Promise((resolve, reject) => {
      const requestId = this.nextRequestId++;
      this.pendingRequests.set(requestId, { resolve, reject });
      this.workers[workerIdx].postMessage({ ...message, requestId });
    });
  }
  
  async createSession(options = {}) {
    if (!this.initialized) await this.init();
    
    const sessionId = this.nextSessionId++;
    
    // Assign to worker with least sessions (round-robin for simplicity)
    const workerIdx = sessionId % this.workers.length;
    this.sessionWorkerMap.set(sessionId, workerIdx);
    
    await this._sendToWorker(workerIdx, {
      type: 'createSession',
      sessionId,
      options,
    });
    
    return sessionId;
  }
  
  async processAudio(sessionId, samples) {
    const workerIdx = this.sessionWorkerMap.get(sessionId);
    if (workerIdx === undefined) {
      throw new Error(`Session ${sessionId} not found`);
    }
    
    const response = await this._sendToWorker(workerIdx, {
      type: 'processAudio',
      sessionId,
      samples: Array.from(samples),
    });
    
    return {
      ...response.result,
      processingTimeMs: response.processingTimeMs,
    };
  }
  
  async flushSession(sessionId) {
    const workerIdx = this.sessionWorkerMap.get(sessionId);
    if (workerIdx === undefined) {
      throw new Error(`Session ${sessionId} not found`);
    }
    
    const response = await this._sendToWorker(workerIdx, {
      type: 'flushSession',
      sessionId,
    });
    
    return response.result;
  }
  
  async closeSession(sessionId) {
    const workerIdx = this.sessionWorkerMap.get(sessionId);
    if (workerIdx === undefined) return;
    
    await this._sendToWorker(workerIdx, {
      type: 'closeSession',
      sessionId,
    });
    
    this.sessionWorkerMap.delete(sessionId);
  }
  
  get sessionCount() {
    return this.sessionWorkerMap.size;
  }
  
  destroy() {
    for (const worker of this.workers) {
      worker.postMessage({ type: 'exit' });
    }
    this.workers = [];
    this.sessionWorkerMap.clear();
    this.pendingRequests.clear();
  }
}

function createStreamingPool(modelPath, options = {}) {
  return new StreamingWorkerPool(modelPath, options);
}

module.exports = {
  StreamingWorkerPool,
  createStreamingPool,
};
