"""
Small Emotiv Cortex integration helper.

Behavior:
- If --emotiv is enabled in your main app, this module exposes EEGThread,
  which attempts to connect to Emotiv Cortex via the WebSocket API and subscribe
  to a simple stream (or falls back to a simulator if not available).

Notes:
- You must supply CLIENT_ID and CLIENT_SECRET (or run simulator mode).
- The module defaults to a simulator when websocket-client is not available
  or connection fails. Do NOT commit secrets into git.
"""
import json
import logging
import threading
import time
from queue import Queue, Empty

try:
    import websocket
    import requests
    WS_AVAILABLE = True
except Exception:
    WS_AVAILABLE = False

class EEGThread(threading.Thread):
    def __init__(self, client_id=None, client_secret=None, use_simulator=False):
        super().__init__(daemon=True)
        self.client_id = client_id
        self.client_secret = client_secret
        # if websocket client not installed or explicit simulator requested, use simulator
        self.use_simulator = use_simulator or not WS_AVAILABLE
        self.running = threading.Event()
        self.running.set()
        self.lock = threading.Lock()
        # latest metrics dictionary (safe to read from main thread)
        self.latest = {
            "engagement": None,
            "stress": None,
            "excitement": None,
            "timestamp": None,
        }
        self._ws = None
        self._queue = Queue(maxsize=1)

    def connect_and_subscribe(self):
        """
        Minimal placeholder for Cortex connect. Replace ws_url and flow as needed
        for your Cortex setup (cloud/local). This will fall back to simulator on failure.
        """
        try:
            if not WS_AVAILABLE:
                raise RuntimeError("websocket-client or requests not installed")
            # Adjust endpoint for your Cortex server. Many local dev setups use wss://localhost:6868
            ws_url = "wss://localhost:6868"
            self._ws = websocket.create_connection(ws_url, timeout=5)
            logging.info("Connected to Cortex websocket at %s", ws_url)
            # NOTE: real Cortex requires auth/subscribe messages; implement per Emotiv docs.
            return True
        except Exception as e:
            logging.warning("Cortex WS connect failed: %s", e)
            self._ws = None
            return False

    def enqueue(self, sample):
        try:
            if self._queue.full():
                _ = self._queue.get_nowait()
            self._queue.put_nowait(sample)
        except Exception:
            pass

    def run(self):
        logging.info("EEGThread starting (simulator=%s)", self.use_simulator)
        if self.use_simulator:
            # produce synthetic metrics every 0.5s
            import random
            while self.running.is_set():
                sample = {
                    "engagement": round(0.3 + 0.7 * random.random(), 3),
                    "stress": round(0.1 + 0.8 * random.random(), 3),
                    "excitement": round(0.2 + 0.7 * random.random(), 3),
                    "timestamp": time.time(),
                }
                with self.lock:
                    self.latest.update(sample)
                time.sleep(0.5)
            return

        # Try connect
        if not self.connect_and_subscribe():
            logging.info("Falling back to simulator because connect failed")
            self.use_simulator = True
            self.run()
            return

        # If connected, read messages
        try:
            while self.running.is_set():
                try:
                    msg = self._ws.recv()
                    if not msg:
                        time.sleep(0.01)
                        continue
                    data = None
                    try:
                        data = json.loads(msg)
                    except Exception:
                        data = None
                    # Data handling depends on the stream you subscribed to
                    if isinstance(data, dict):
                        sample = {}
                        # Example mapping - adjust to actual payload keys
                        for k in ("engagement", "stress", "excitement"):
                            if k in data:
                                try:
                                    sample[k] = float(data[k])
                                except Exception:
                                    sample[k] = data[k]
                        sample["timestamp"] = time.time()
                        if sample:
                            with self.lock:
                                self.latest.update(sample)
                except Exception as e:
                    logging.exception("EEGThread recv error: %s", e)
                    time.sleep(0.5)
        finally:
            try:
                if self._ws:
                    self._ws.close()
            except Exception:
                pass

    def stop(self):
        self.running.clear()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    def get_latest(self):
        with self.lock:
            return dict(self.latest)
