const express = require('express');
const axios = require('axios');
const app = express();
app.use(express.json());
app.post('/sensor', (req, res) => { console.log('Received sensor payload:', req.body); res.json({ status: 'ok' }); });
async function emitSimulatedSensor() {
  const payload = {
    timestamp: new Date().toISOString(),
    camera: { detections: [{ x: 10, y: 10, w: 40, h: 30, label: 'blob' }] },
    eeg: Array.from({length:256}, (_,i) => Math.sin(2*Math.PI*10*(i/256)) + (Math.random()-0.5)),
    ecg: Array.from({length:256}, (_,i) => Math.sin(2*Math.PI*1.2*(i/256)) + (Math.random()-0.5)),
    gps: { lat: 0.0, lon: 0.0, fix: false }
  };
  try { await axios.post('http://localhost:3000/sensor', payload, { timeout: 2000 }); } catch (e) { console.log('Emit failed:', e.message); }
}
const EMIT_INTERVAL_MS = 5000;
app.listen(3000, () => { console.log('Node example server listening on http://localhost:3000'); setInterval(emitSimulatedSensor, EMIT_INTERVAL_MS); });
