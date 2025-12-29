#!/usr/bin/env node
/**
 * Sample transcription server for stress testing
 * 
 * Run with: node stress-tests/server.js
 * 
 * Endpoints:
 *   POST /transcribe - Transcribe audio file (multipart/form-data)
 *   POST /transcribe-buffer - Transcribe audio buffer (raw body)
 *   GET /health - Health check
 *   GET /stats - Server statistics
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// Load the engine
const { Engine, decodeAudio } = require('../index');

const PORT = process.env.PORT || 3000;
const MODEL_PATH = process.env.MODEL_PATH || './models/tiny';

// Check model exists
if (!fs.existsSync(path.join(MODEL_PATH, 'model.bin'))) {
  console.error('❌ Model not found at:', MODEL_PATH);
  console.error('   Download with: cd models && git lfs install && git clone --depth 1 https://huggingface.co/Systran/faster-whisper-tiny tiny');
  process.exit(1);
}

// Statistics
const stats = {
  totalRequests: 0,
  successfulRequests: 0,
  failedRequests: 0,
  activeRequests: 0,
  peakActiveRequests: 0,
  totalProcessingTimeMs: 0,
  startTime: Date.now(),
};

// Create engine pool for concurrent requests
const POOL_SIZE = parseInt(process.env.POOL_SIZE || '4');
const engines = [];
const engineLocks = [];

console.log(`Loading ${POOL_SIZE} engine instances...`);
for (let i = 0; i < POOL_SIZE; i++) {
  engines.push(new Engine(MODEL_PATH));
  engineLocks.push(false);
  console.log(`  Engine ${i + 1}/${POOL_SIZE} loaded`);
}

// Get an available engine (simple round-robin with waiting)
async function getEngine() {
  while (true) {
    for (let i = 0; i < engines.length; i++) {
      if (!engineLocks[i]) {
        engineLocks[i] = true;
        return { engine: engines[i], index: i };
      }
    }
    // Wait a bit and try again
    await new Promise(resolve => setTimeout(resolve, 10));
  }
}

function releaseEngine(index) {
  engineLocks[index] = false;
}

// Parse multipart form data (simple implementation)
function parseMultipart(buffer, boundary) {
  const parts = {};
  const boundaryBuffer = Buffer.from(`--${boundary}`);
  
  let start = 0;
  let pos = buffer.indexOf(boundaryBuffer, start);
  
  while (pos !== -1) {
    const nextPos = buffer.indexOf(boundaryBuffer, pos + boundaryBuffer.length);
    if (nextPos === -1) break;
    
    const part = buffer.slice(pos + boundaryBuffer.length, nextPos);
    const headerEnd = part.indexOf('\r\n\r\n');
    if (headerEnd === -1) {
      pos = nextPos;
      continue;
    }
    
    const headers = part.slice(0, headerEnd).toString();
    const content = part.slice(headerEnd + 4, -2); // Remove trailing \r\n
    
    const nameMatch = headers.match(/name="([^"]+)"/);
    if (nameMatch) {
      const name = nameMatch[1];
      parts[name] = content;
    }
    
    pos = nextPos;
  }
  
  return parts;
}

// Request handler
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }
  
  // Health check
  if (url.pathname === '/health' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', engines: POOL_SIZE }));
    return;
  }
  
  // Stats endpoint
  if (url.pathname === '/stats' && req.method === 'GET') {
    const uptime = Date.now() - stats.startTime;
    const avgProcessingTime = stats.successfulRequests > 0 
      ? stats.totalProcessingTimeMs / stats.successfulRequests 
      : 0;
    
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      ...stats,
      uptimeMs: uptime,
      avgProcessingTimeMs: avgProcessingTime.toFixed(2),
      requestsPerSecond: (stats.totalRequests / (uptime / 1000)).toFixed(2),
    }));
    return;
  }
  
  // Transcribe endpoint (file path in query or body)
  if (url.pathname === '/transcribe' && req.method === 'POST') {
    stats.totalRequests++;
    stats.activeRequests++;
    stats.peakActiveRequests = Math.max(stats.peakActiveRequests, stats.activeRequests);
    
    const startTime = Date.now();
    let engineIndex = -1;
    
    try {
      // Read request body
      const chunks = [];
      for await (const chunk of req) {
        chunks.push(chunk);
      }
      const body = Buffer.concat(chunks);
      
      let audioPath = null;
      let audioBuffer = null;
      let options = {};
      
      // Check content type
      const contentType = req.headers['content-type'] || '';
      
      if (contentType.includes('multipart/form-data')) {
        // Parse multipart
        const boundary = contentType.split('boundary=')[1];
        const parts = parseMultipart(body, boundary);
        
        if (parts.file) {
          audioBuffer = parts.file;
        }
        if (parts.options) {
          options = JSON.parse(parts.options.toString());
        }
      } else if (contentType.includes('application/json')) {
        // JSON body with file path
        const json = JSON.parse(body.toString());
        audioPath = json.path;
        options = json.options || {};
      } else {
        // Raw audio buffer
        audioBuffer = body;
      }
      
      // Also check query param
      if (!audioPath && !audioBuffer) {
        audioPath = url.searchParams.get('path');
      }
      
      if (!audioPath && !audioBuffer) {
        throw new Error('No audio provided. Send file path in JSON body or audio buffer.');
      }
      
      // Get engine from pool
      const { engine, index } = await getEngine();
      engineIndex = index;
      
      // Transcribe
      let result;
      if (audioBuffer) {
        result = engine.transcribeBuffer(audioBuffer, options);
      } else {
        result = engine.transcribeFile(audioPath, options);
      }
      
      const processingTime = Date.now() - startTime;
      stats.successfulRequests++;
      stats.totalProcessingTimeMs += processingTime;
      
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        success: true,
        processingTimeMs: processingTime,
        result,
      }));
      
    } catch (err) {
      stats.failedRequests++;
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        success: false,
        error: err.message,
      }));
    } finally {
      stats.activeRequests--;
      if (engineIndex >= 0) {
        releaseEngine(engineIndex);
      }
    }
    return;
  }
  
  // 404
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

server.listen(PORT, () => {
  console.log(`\n🚀 Transcription server running on http://localhost:${PORT}`);
  console.log(`   Engine pool size: ${POOL_SIZE}`);
  console.log(`\nEndpoints:`);
  console.log(`   POST /transcribe - Transcribe audio`);
  console.log(`   GET /health - Health check`);
  console.log(`   GET /stats - Server statistics`);
  console.log(`\nPress Ctrl+C to stop\n`);
});
