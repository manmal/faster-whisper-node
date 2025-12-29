/**
 * StreamingBatcher - Batched concurrent transcription for multiple streams
 * 
 * This module provides high-throughput concurrent transcription by:
 * 1. Collecting audio chunks from multiple streams
 * 2. Processing them in batches to maximize throughput
 * 3. Using a single Engine instance to avoid memory duplication
 * 
 * Key design: Instead of each stream having its own Engine, we share one Engine
 * and batch requests together. This dramatically improves throughput for many
 * concurrent streams.
 */

const { Engine, decodeAudio } = require('./index.js');
const { EventEmitter } = require('events');

/**
 * Configuration options for the StreamingBatcher
 * @typedef {Object} BatcherOptions
 * @property {number} [maxBatchSize=8] - Maximum number of chunks to process in one batch
 * @property {number} [maxWaitMs=50] - Maximum time to wait before processing a partial batch
 * @property {number} [numWorkers=4] - Number of worker threads for parallel processing
 * @property {string} [language='en'] - Language code for transcription
 * @property {number} [beamSize=5] - Beam size for transcription
 * @property {boolean} [wordTimestamps=false] - Include word-level timestamps
 */

/**
 * Represents a pending transcription request
 * @typedef {Object} PendingRequest
 * @property {number} streamId - ID of the stream
 * @property {number} chunkId - ID of the chunk within the stream
 * @property {Float32Array} samples - Audio samples
 * @property {number} audioStartS - Start time in original audio
 * @property {number} audioDurationS - Duration of the chunk
 * @property {Function} resolve - Promise resolve function
 * @property {Function} reject - Promise reject function
 * @property {number} submitTime - Time when request was submitted
 */

class StreamingBatcher extends EventEmitter {
  /**
   * Create a new StreamingBatcher
   * @param {Engine} engine - The transcription engine
   * @param {BatcherOptions} [options] - Configuration options
   */
  constructor(engine, options = {}) {
    super();
    
    this.engine = engine;
    this.options = {
      maxBatchSize: options.maxBatchSize || 8,
      maxWaitMs: options.maxWaitMs || 50,
      language: options.language || 'en',
      beamSize: options.beamSize || 5,
      wordTimestamps: options.wordTimestamps || false,
    };
    
    /** @type {PendingRequest[]} */
    this.pendingQueue = [];
    
    /** @type {boolean} */
    this.isProcessing = false;
    
    /** @type {NodeJS.Timeout|null} */
    this.batchTimer = null;
    
    /** @type {number} */
    this.nextStreamId = 0;
    
    /** @type {Map<number, StreamContext>} */
    this.streams = new Map();
    
    // Stats
    this.stats = {
      totalChunks: 0,
      totalBatches: 0,
      totalProcessingMs: 0,
      queueHighWaterMark: 0,
    };
  }
  
  /**
   * Create a new stream and return its ID
   * @returns {number} Stream ID
   */
  createStream() {
    const streamId = this.nextStreamId++;
    this.streams.set(streamId, {
      id: streamId,
      chunkCount: 0,
      totalAudioS: 0,
      createdAt: Date.now(),
    });
    return streamId;
  }
  
  /**
   * Close a stream
   * @param {number} streamId - Stream ID to close
   */
  closeStream(streamId) {
    this.streams.delete(streamId);
  }
  
  /**
   * Submit an audio chunk for transcription
   * @param {number} streamId - Stream ID
   * @param {Float32Array|number[]} samples - Audio samples (16kHz mono, -1 to 1)
   * @param {number} [audioStartS=0] - Start time in original audio
   * @returns {Promise<Object>} Transcription result
   */
  async transcribeChunk(streamId, samples, audioStartS = 0) {
    const stream = this.streams.get(streamId);
    if (!stream) {
      throw new Error(`Stream ${streamId} not found`);
    }
    
    const chunkId = stream.chunkCount++;
    const samplesArray = samples instanceof Float32Array ? samples : new Float32Array(samples);
    const audioDurationS = samplesArray.length / 16000;
    stream.totalAudioS += audioDurationS;
    
    return new Promise((resolve, reject) => {
      const request = {
        streamId,
        chunkId,
        samples: samplesArray,
        audioStartS,
        audioDurationS,
        resolve,
        reject,
        submitTime: Date.now(),
      };
      
      this.pendingQueue.push(request);
      this.stats.queueHighWaterMark = Math.max(this.stats.queueHighWaterMark, this.pendingQueue.length);
      
      this._scheduleProcessing();
    });
  }
  
  /**
   * Schedule batch processing
   * @private
   */
  _scheduleProcessing() {
    // If we have enough for a full batch, process immediately
    if (this.pendingQueue.length >= this.options.maxBatchSize && !this.isProcessing) {
      this._processNextBatch();
      return;
    }
    
    // Otherwise, set a timer if not already set
    if (!this.batchTimer && !this.isProcessing) {
      this.batchTimer = setTimeout(() => {
        this.batchTimer = null;
        if (this.pendingQueue.length > 0 && !this.isProcessing) {
          this._processNextBatch();
        }
      }, this.options.maxWaitMs);
    }
  }
  
  /**
   * Process the next batch of requests
   * @private
   */
  async _processNextBatch() {
    if (this.isProcessing || this.pendingQueue.length === 0) {
      return;
    }
    
    this.isProcessing = true;
    
    // Take up to maxBatchSize requests
    const batch = this.pendingQueue.splice(0, this.options.maxBatchSize);
    const batchStartTime = Date.now();
    
    this.emit('batchStart', { size: batch.length });
    
    try {
      // Process each chunk in the batch
      // TODO: In the future, we could implement true batched inference at the Rust level
      // For now, we process sequentially but benefit from shared Engine
      const results = await Promise.all(
        batch.map(async (request) => {
          try {
            const processStart = Date.now();
            
            // Convert Float32Array to regular array for NAPI
            const samplesArray = Array.from(request.samples);
            
            const result = this.engine.transcribeSamples(samplesArray, {
              language: this.options.language,
              beamSize: this.options.beamSize,
              wordTimestamps: this.options.wordTimestamps,
            });
            
            const processEnd = Date.now();
            
            return {
              request,
              result: {
                streamId: request.streamId,
                chunkId: request.chunkId,
                audioStartS: request.audioStartS,
                audioDurationS: request.audioDurationS,
                text: result.text,
                segments: result.segments,
                processingTimeMs: processEnd - processStart,
                queueTimeMs: processStart - request.submitTime,
              },
              error: null,
            };
          } catch (error) {
            return {
              request,
              result: null,
              error,
            };
          }
        })
      );
      
      // Resolve/reject all promises
      for (const { request, result, error } of results) {
        if (error) {
          request.reject(error);
        } else {
          request.resolve(result);
        }
      }
      
      const batchEndTime = Date.now();
      const batchDurationMs = batchEndTime - batchStartTime;
      
      this.stats.totalChunks += batch.length;
      this.stats.totalBatches++;
      this.stats.totalProcessingMs += batchDurationMs;
      
      this.emit('batchComplete', {
        size: batch.length,
        durationMs: batchDurationMs,
        avgPerChunkMs: batchDurationMs / batch.length,
      });
      
    } catch (error) {
      // Reject all pending requests in this batch
      for (const request of batch) {
        request.reject(error);
      }
      
      this.emit('batchError', { size: batch.length, error });
    } finally {
      this.isProcessing = false;
      
      // Process next batch if queue has more items
      if (this.pendingQueue.length > 0) {
        setImmediate(() => this._scheduleProcessing());
      }
    }
  }
  
  /**
   * Wait for all pending requests to complete
   * @returns {Promise<void>}
   */
  async flush() {
    while (this.pendingQueue.length > 0 || this.isProcessing) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
  }
  
  /**
   * Get current statistics
   * @returns {Object} Statistics
   */
  getStats() {
    return {
      ...this.stats,
      pendingRequests: this.pendingQueue.length,
      activeStreams: this.streams.size,
      avgBatchSize: this.stats.totalBatches > 0 
        ? this.stats.totalChunks / this.stats.totalBatches 
        : 0,
      avgProcessingMsPerChunk: this.stats.totalChunks > 0
        ? this.stats.totalProcessingMs / this.stats.totalChunks
        : 0,
    };
  }
  
  /**
   * Destroy the batcher and cleanup resources
   */
  destroy() {
    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.batchTimer = null;
    }
    
    // Reject all pending requests
    for (const request of this.pendingQueue) {
      request.reject(new Error('Batcher destroyed'));
    }
    this.pendingQueue = [];
    this.streams.clear();
  }
}

/**
 * Create a StreamingBatcher with a new Engine
 * @param {string} modelPath - Path to the model
 * @param {BatcherOptions} [options] - Batcher options
 * @returns {StreamingBatcher}
 */
function createBatcher(modelPath, options = {}) {
  const engine = new Engine(modelPath);
  return new StreamingBatcher(engine, options);
}

module.exports = {
  StreamingBatcher,
  createBatcher,
};
