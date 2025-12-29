const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const path = require('path');

if (!isMainThread) {
  const { Engine, decodeAudio } = require(workerData.enginePath);
  const engine = new Engine(workerData.modelPath);
  const samples = decodeAudio(workerData.audioPath);
  const samples2s = Array.from(samples.slice(0, 32000));
  
  parentPort.on('message', (msg) => {
    if (msg.type === 'test') {
      const start = Date.now();
      engine.transcribeSamples(samples2s, { language: 'en', beamSize: 5 });
      parentPort.postMessage({ type: 'done', time: Date.now() - start });
    }
  });
  
  parentPort.postMessage({ type: 'ready' });
} else {
  async function test() {
    const numWorkers = 8;
    const workers = [];
    
    console.log('Creating', numWorkers, 'workers...');
    
    // Create workers
    for (let i = 0; i < numWorkers; i++) {
      const worker = new Worker(__filename, {
        workerData: {
          enginePath: path.resolve(__dirname, '../index.js'),
          modelPath: path.resolve(__dirname, '../models/tiny'),
          audioPath: path.resolve(__dirname, '../tests/fixtures/hello.wav'),
        }
      });
      
      await new Promise((resolve) => {
        worker.once('message', (msg) => {
          if (msg.type === 'ready') resolve();
        });
      });
      
      workers.push(worker);
    }
    
    console.log('All', numWorkers, 'workers ready\n');
    
    // Sequential test
    console.log('Sequential (one at a time)...');
    const seqStart = Date.now();
    for (let i = 0; i < 16; i++) {
      await new Promise((resolve) => {
        workers[i % numWorkers].once('message', (msg) => resolve(msg.time));
        workers[i % numWorkers].postMessage({ type: 'test' });
      });
    }
    console.log('Sequential: 16 chunks in', Date.now() - seqStart, 'ms (' + ((Date.now() - seqStart)/16).toFixed(0) + 'ms each)');
    
    // Parallel test  
    console.log('\nParallel (all workers at once)...');
    const parStart = Date.now();
    for (let round = 0; round < 2; round++) {
      await Promise.all(workers.map((worker) => {
        return new Promise((resolve) => {
          worker.once('message', (msg) => resolve(msg.time));
          worker.postMessage({ type: 'test' });
        });
      }));
    }
    console.log('Parallel: 16 chunks (2 rounds x', numWorkers, 'workers) in', Date.now() - parStart, 'ms (' + ((Date.now() - parStart)/16).toFixed(0) + 'ms each)');
    
    // Heavy parallel test
    console.log('\nHeavy parallel (100 chunks distributed)...');
    const heavyStart = Date.now();
    const promises = [];
    for (let i = 0; i < 100; i++) {
      const workerIdx = i % numWorkers;
      promises.push(new Promise((resolve) => {
        const handler = (msg) => {
          if (msg.type === 'done') {
            workers[workerIdx].off('message', handler);
            resolve(msg.time);
          }
        };
        workers[workerIdx].on('message', handler);
        workers[workerIdx].postMessage({ type: 'test' });
      }));
    }
    const times = await Promise.all(promises);
    console.log('Heavy: 100 chunks in', Date.now() - heavyStart, 'ms');
    console.log('  Avg processing time:', (times.reduce((a,b)=>a+b,0)/times.length).toFixed(0), 'ms');
    console.log('  Min:', Math.min(...times), 'ms, Max:', Math.max(...times), 'ms');
    console.log('  Throughput:', (100 * 2 / ((Date.now() - heavyStart)/1000)).toFixed(1), 'seconds of audio per second');
    
    // Cleanup
    for (const worker of workers) {
      worker.terminate();
    }
  }
  
  test().catch(console.error);
}
