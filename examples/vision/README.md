# YOLO Unified Live (examples/vision)

This folder contains a unified live viewer that runs
YOLO (ultralytics) on a webcam or RealSense stream, optionally computes
MiDaS relative depth, and can ingest Emotiv Insight EEG metrics.

Quick features
- Live YOLO detections (ultralytics/yolov8)
- Optional MiDaS depth (background thread or every-N-frames)
- Optional Emotiv Insight (EEGThread) — simulator fallback if no device
- Save per-detection crops and a CSV log with detection + EEG fields
- Keyboard controls for quick toggles

Quick start (recommended)
1. Activate your venv:
   source .venv/bin/activate

2. Test the viewer (fast, no MiDaS) with simulated EEG:
   export PYTHONPATH="$PWD:$PYTHONPATH"
   export QT_QPA_PLATFORM=xcb
   export DISPLAY=:0
   python3 -u examples/vision/webcam_yolo_unified_threaded.py --device cpu --emotiv

3. To save CSV and crops:
   python3 -u examples/vision/webcam_yolo_unified_threaded.py --device cpu --emotiv --save-csv --out live_detections.csv --save-crops --crops-dir data/live_crops

4. To enable MiDaS in background thread (slow on CPU):
   python3 -u examples/vision/webcam_yolo_unified_threaded.py --device cpu --midas --midas-thread --emotiv --save-csv --out live_detections.csv

Emotiv / Cortex notes
- Do NOT put client id/secret in git. Use environment variables or CLI flags:
  export EMOTIV_CLIENT_ID="..." && export EMOTIV_CLIENT_SECRET="..."
  python3 examples/vision/webcam_yolo_unified_threaded.py --emotiv --emotiv-client-id "$EMOTIV_CLIENT_ID" --emotiv-client-secret "$EMOTIV_CLIENT_SECRET"
- If Cortex (wss://localhost:6868) or your Emotiv service is not available, the EEGThread falls back to a simulator to generate synthetic metrics for UI/testing.

Keyboard controls (when the viewer window is focused)
- ESC — quit
- s — save screenshot of the combined view
- c — toggle saving crops
- m — toggle MiDaS (loads weights on first enable)
- p — pause/resume capture
- h — help (prints keys to side panel)

Troubleshooting
- If OpenCV / Qt reports Wayland errors, try:
  export QT_QPA_PLATFORM=xcb
  or if you run under Wayland, try QT_QPA_PLATFORM=wayland
- If imports fail, ensure you run from repo root and set:
  export PYTHONPATH="$PWD:$PYTHONPATH"
- To check Python syntax locally:
  python3 -m py_compile examples/vision/webcam_yolo_unified_threaded.py examples/vision/emotiv_integration.py

License & privacy
- EEG data is sensitive. Do not publish raw EEG or credentials without consent.
