#!/usr/bin/env python3
# (#!/usr/bin/env python3
"""
Unified YOLO live viewer with non-blocking MiDaS and robust logging.

Features:
- YOLO live display + right-side panel (thumbnails + MiDaS heatmap)
- Optional RealSense metric depth
- Save crops + CSV
- Two non-blocking MiDaS modes:
    --midas-every N   : compute MiDaS only once every N frames (default N=5)
    --midas-thread    : run MiDaS in a background thread (preferred for UI responsiveness)
- Robust logging to ~/yolo_unified_run.log and crash traces to ~/yolo_unified_crash.log
- Falls back to headless capture (no GUI) when imshow fails
"""
import argparse
import csv
import logging
import os
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from queue import Queue, Empty

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from examples.vision.emotiv_integration import EEGThread

# RealSense optional
RS_AVAILABLE = True
try:
    import pyrealsense2 as rs
except Exception:
    RS_AVAILABLE = False

LOGFILE = os.path.expanduser("~/yolo_unified_run.log")
CRASHLOG = os.path.expanduser("~/yolo_unified_crash.log")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOGFILE, mode="a")])

MIDAS_MODEL_NAME = "MiDaS_small"


def load_midas_model(device):
    logging.info("Loading MiDaS model (this may download weights)...")
    model = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_NAME)
    model.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform if MIDAS_MODEL_NAME == "MiDaS_small" else transforms.default_transform
    return model, transform


def normalize_depth_for_display(depth_np, out_h, out_w):
    d = depth_np.copy().astype(np.float32)
    d = d - np.nanmin(d)
    mx = np.nanmax(d)
    if mx > 0:
        d = d / mx
    d8 = (255.0 * d).astype(np.uint8)
    d8 = cv2.resize(d8, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    cmap = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    return cmap


def draw_label_box(img, x1, y1, x2, y2, label, color=(16, 200, 64), thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def make_side_panel(depth_cmap, thumbnails, panel_w, panel_h, log_lines):
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8) + 16
    y = 10
    pad = 8
    if depth_cmap is not None:
        dh = min(depth_cmap.shape[0], panel_h // 3)
        dw = min(depth_cmap.shape[1], panel_w - 2 * pad)
        small = cv2.resize(depth_cmap, (dw, dh))
        panel[y:y + dh, pad:pad + dw] = small
        cv2.putText(panel, "MiDaS depth", (pad, y + dh + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        y += dh + 32
    else:
        cv2.putText(panel, "MiDaS: off", (pad, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        y += 28

    if thumbnails:
        thumb_h = max(40, (panel_h - y - 100) // 4)
        thumb_w = panel_w - 2 * pad
        for (label, thumb) in thumbnails:
            if y + thumb_h + 4 > panel_h - 80:
                break
            if thumb is None:
                timg = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8) + 40
            else:
                timg = cv2.resize(thumb, (thumb_w, thumb_h))
            panel[y:y + thumb_h, pad:pad + thumb_w] = timg
            cv2.putText(panel, label[:18], (pad + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += thumb_h + 8

    y_log = panel_h - 20 * len(log_lines) - 12
    if y_log < y:
        y_log = y + 8
    for i, ln in enumerate(log_lines[-6:]):
        cv2.putText(panel, ln[:60], (pad, y_log + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    cv2.putText(panel, "ESC:quit s:save c:toggle crops m:toggle MiDaS p:pause", (pad, panel_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)
    return panel


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


class MidasThread(threading.Thread):
    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self.model = None
        self.transform = None
        self._queue = Queue(maxsize=1)  # keep only latest frame
        self.depth_map = None
        self.lock = threading.Lock()
        self.running = threading.Event()
        self.running.set()

    def load(self):
        try:
            self.model, self.transform = load_midas_model(self.device)
        except Exception:
            logging.exception("Failed to load MiDaS in thread")
            self.model = None

    def enqueue(self, frame_rgb):
        # replace latest
        try:
            if self._queue.full():
                _ = self._queue.get_nowait()
            self._queue.put_nowait(frame_rgb)
        except Exception:
            pass

    def run(self):
        logging.info("MiDaS thread starting")
        if self.model is None:
            self.load()
        while self.running.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if frame is None:
                continue
            try:
                with torch.no_grad():
                    inp = self.transform(frame).to(self.device)
                    pred = self.model(inp)
                    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=frame.shape[:2],
                                                           mode="bilinear", align_corners=False).squeeze()
                    d = pred.cpu().numpy()
                with self.lock:
                    self.depth_map = d
            except Exception:
                logging.exception("MiDaS thread inference failed")
        logging.info("MiDaS thread exiting")

    def stop(self):
        self.running.clear()
        # wake queue
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass


def run_unified(args):
    # logging startup info
    logging.info("Starting unified viewer")
    logging.info("Args: %s", args)

    # load YOLO
    model = YOLO(args.model)

    # MiDaS setup
    midas_thread = None
    midas_model = None
    midas_transform = None
    use_midas_thread = args.midas_thread
    midas_every = int(args.midas_every) if args.midas_every else 5

    if args.midas and not use_midas_thread and not args.midas_every:
        # default single-thread fallback: load model in main thread
        try:
            midas_model, midas_transform = load_midas_model(args.device)
        except Exception:
            logging.exception("MiDaS failed to load in main thread; continuing without MiDaS")
            midas_model = None

    if args.midas and use_midas_thread:
        midas_thread = MidasThread(args.device)
        # load lazily inside thread when it starts
        midas_thread.load()
        midas_thread.start()

    pipeline = None
    align = None
    depth_scale = None
    color_intr = None
    cap = None

    if args.realsense:
        if not RS_AVAILABLE:
            raise SystemExit("pyrealsense2 not installed")
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(cfg)
        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        color_stream = profile.get_stream(rs.stream.color)
        color_intr = color_stream.as_video_stream_profile().get_intrinsics()
        logging.info("RealSense depth_scale=%s intr=%s", depth_scale, color_intr)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"ERROR: camera {args.camera} not available")

    if args.save_crops:
        safe_mkdir(args.crops_dir)

    csv_file = None
    csv_writer = None
    if args.save_csv:
        out_fn = args.out if args.out else "live_detections.csv"
        csv_file = open(out_fn, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "label", "conf", "x1", "y1", "x2", "y2", "cx", "cy",
                             "depth_rel", "depth_m", "xyz_m", "crop_path", "e_engagement", "e_stress", "e_excitement"])

    recent_thumbs = deque(maxlen=12)
    log_lines = deque(maxlen=8)

    paused = False
    save_crops_toggle = args.save_crops
    frame_idx = 0

    # attempt GUI, but fallback to headless if fails
    gui_ok = True
    try:
        cv2.namedWindow("YOLO Unified Live", cv2.WINDOW_NORMAL)
    except Exception:
        logging.exception("cv2.namedWindow failed; falling back to headless mode")
        gui_ok = False

    try:
        while True:
            if paused:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                time.sleep(0.05)
                continue

            frame_idx += 1
            if args.realsense:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    log_lines.append("No RealSense frames")
                    time.sleep(0.01)
                    continue
                color = np.asanyarray(color_frame.get_data())
                depth_raw = np.asanyarray(depth_frame.get_data())
                depth_m = depth_raw.astype(np.float32) * depth_scale
            else:
                ret, color = cap.read()
                if not ret:
                    log_lines.append(f"Frame grab failed at {frame_idx}")
                    time.sleep(0.05)
                    continue
                depth_m = None

            # MiDaS: either background thread or every-N-frames strategy
            depth_map = None
            depth_cmap = None
            if args.midas:
                if midas_thread is not None:
                    # enqueue RGB for background processing (non-blocking)
                    try:
                        midas_thread.enqueue(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
                        with midas_thread.lock:
                            depth_map = midas_thread.depth_map.copy() if midas_thread.depth_map is not None else None
                    except Exception:
                        depth_map = None
                elif midas_model is not None:
                    # compute in main thread only every N frames
                    if (frame_idx % midas_every) == 0:
                        try:
                            img_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                            input_t = midas_transform(img_rgb).to(args.device)
                            with torch.no_grad():
                                pred = midas_model(input_t)
                                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=img_rgb.shape[:2],
                                                                       mode="bilinear", align_corners=False).squeeze()
                                depth_map = pred.cpu().numpy()
                        except Exception:
                            logging.exception("MiDaS inference failed in main thread")
                            depth_map = None

            if depth_map is not None:
                try:
                    depth_cmap = normalize_depth_for_display(depth_map, out_h=200, out_w=args.panel_width - 16)
                except Exception:
                    depth_cmap = None

            # YOLO
            start = time.time()
            results = model(color, imgsz=640, device=args.device, conf=args.conf)
            elapsed = (time.time() - start) * 1000.0

            disp = color.copy()
            detections_this_frame = 0

            for res in results:
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy().reshape(-1)
                    classes = boxes.cls.cpu().numpy().astype(int).reshape(-1)
                except Exception:
                    xyxy = [b.xyxy[0].numpy() for b in boxes]
                    confs = [float(b.conf[0]) for b in boxes]
                    classes = [int(b.cls[0]) if hasattr(b, "cls") else -1 for b in boxes]

                for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    label_name = res.names[int(cls)] if hasattr(res, "names") else str(cls)
                    label_text = f"{label_name} {conf:.2f}"

                    depth_rel_val = ""
                    if depth_map is not None:
                        vy1, vy2 = max(0, cy - 2), min(depth_map.shape[0], cy + 3)
                        vx1, vx2 = max(0, cx - 2), min(depth_map.shape[1], cx + 3)
                        window = depth_map[vy1:vy2, vx1:vx2]
                        if window.size:
                            depth_rel_val = float(np.median(window))
                            label_text += f" d={depth_rel_val:.3f}"

                    depth_m_val = ""
                    xyz_m = ""
                    if args.realsense and depth_m is not None:
                        wy1, wy2 = max(0, cy - 2), min(depth_m.shape[0], cy + 3)
                        wx1, wx2 = max(0, cx - 2), min(depth_m.shape[1], cx + 3)
                        win = depth_m[wy1:wy2, wx1:wx2]
                        if win.size:
                            z = float(np.median(win))
                            depth_m_val = f"{z:.4f}"
                            xyz_m = ""  # compute if intrinsics available

                    draw_label_box(disp, x1, y1, x2, y2, label_text)

                    crop_path = ""
                    if save_crops_toggle:
                        h, w = color.shape[:2]
                        x1c = max(0, min(w - 1, x1))
                        y1c = max(0, min(h - 1, y1))
                        x2c = max(0, min(w - 1, x2))
                        y2c = max(0, min(h - 1, y2))
                        if x2c <= x1c or y2c <= y1c:
                            crop_w = min(64, w)
                            crop_h = min(64, h)
                            x1c = max(0, cx - crop_w // 2)
                            y1c = max(0, cy - crop_h // 2)
                            x2c = min(w, x1c + crop_w)
                            y2c = min(h, y1c + crop_h)
                        crop = color[y1c:y2c, x1c:x2c]
                        label_dir = os.path.join(args.crops_dir, label_name)
                        safe_mkdir(label_dir)
                        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        fname = f"{label_name}_{ts_str}_{frame_idx:05d}_{cx}_{cy}.jpg"
                        crop_path = os.path.join(label_dir, fname)
                        try:
                            cv2.imwrite(crop_path, crop)
                            thumb = cv2.resize(crop, (args.panel_width - 16, 64))
                            recent_thumbs.appendleft((label_name, thumb))
                        except Exception:
                            logging.exception("Failed to write crop")
                            crop_path = ""

                    if csv_writer:
                        eeg = emotiv_thread.get_latest() if emotiv_thread else {}
                        csv_writer.writerow([time.time(), label_name, f"{conf:.4f}", x1, y1, x2, y2, cx, cy,
                                             depth_rel_val if depth_rel_val != "" else "",
                                             depth_m_val, xyz_m, crop_path, eeg.get("engagement",""), eeg.get("stress",""), eeg.get("excitement","")])
                        csv_file.flush()

                    detections_this_frame += 1

            # include latest EEG metrics if available
            eeg = emotiv_thread.get_latest() if emotiv_thread else {}
            try:
                e_eng = eeg.get("engagement", "")
                e_str = eeg.get("stress", "")
                e_exc = eeg.get("excitement", "")
                log_lines.append(f"EEG eng={e_eng} stress={e_str} exc={e_exc}")
            except Exception:
                pass

            log_lines.append(f"Frame {frame_idx} det={detections_this_frame} t={elapsed:.1f}ms")
            panel_h = disp.shape[0]
            panel_w = args.panel_width
            side_panel = make_side_panel(depth_cmap, list(recent_thumbs), panel_w, panel_h, list(log_lines))
            combined = np.hstack((cv2.resize(disp, (disp.shape[1], disp.shape[0])), side_panel))

            if gui_ok:
                try:
                    cv2.imshow("YOLO Unified Live", combined)
                except Exception:
                    logging.exception("cv2.imshow failed; switching to headless mode")
                    gui_ok = False
            if gui_ok:
                key = cv2.waitKey(1) & 0xFF
            else:
                # no GUI: emulate pause key checks via stdin (non-blocking)
                key = 0

            if key == 27:
                break
            elif key == ord("s"):
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                fname = f"yolo_unified_{ts}.jpg"
                cv2.imwrite(fname, combined)
                log_lines.append(f"Saved screenshot: {fname}")
            elif key == ord("c"):
                save_crops_toggle = not save_crops_toggle
                log_lines.append(f"Save crops: {'ON' if save_crops_toggle else 'OFF'}")
            elif key == ord("m"):
                if args.midas and midas_thread is None and midas_model is None:
                    # user toggled midas on at runtime; load main-thread midas
                    try:
                        midas_model, midas_transform = load_midas_model(args.device)
                        log_lines.append("MiDaS enabled (main thread)")
                    except Exception:
                        logging.exception("MiDaS enable failed")
                elif args.midas and midas_thread is None and midas_model is not None:
                    midas_model = None
                    midas_transform = None
                    log_lines.append("MiDaS disabled (main thread)")
                elif args.midas and midas_thread is not None:
                    # toggle thread-based midas by starting/stopping thread (not implemented toggle)
                    log_lines.append("MiDaS thread mode active")
            elif key == ord("p"):
                paused = not paused
                log_lines.append(f"Paused: {paused}")
            elif key == ord("h"):
                log_lines.append("Keys: ESC quit | s save | c toggle crops | m toggle MiDaS | p pause")

            if args.frames and frame_idx >= args.frames:
                break

    except Exception as e:
        logging.exception("Unhandled exception in main loop")
        with open(CRASHLOG, "a") as f:
            f.write(datetime.utcnow().isoformat() + " CRASH:\n")
            traceback.print_exc(file=f)
    finally:
        logging.info("Shutting down")
        if cap is not None:
            cap.release()
        if pipeline is not None:
            pipeline.stop()
        if csv_file:
            csv_file.close()
        if midas_thread is not None:
            midas_thread.stop()
            midas_thread.join(timeout=1)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotiv", action="store_true", help="enable Emotiv Insight integration (simulator fallback)")
    ap.add_argument("--emotiv-client-id", default="", help="Emotiv Cortex client id (optional)")
    ap.add_argument("--emotiv-client-secret", default="", help="Emotiv Cortex client secret (optional)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="yolov8n")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--out", default="live_detections.csv")
    ap.add_argument("--midas", action="store_true")
    ap.add_argument("--midas-every", type=int, default=5,
                    help="If set and not using --midas-thread, run MiDaS every N frames")
    ap.add_argument("--midas-thread", action="store_true",
                    help="Run MiDaS in a background thread (preferred for UI)")
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--crops-dir", default="data/crops")
    ap.add_argument("--save-csv", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--panel-width", type=int, default=360)
    args = ap.parse_args()

    # write quick startup banner
    logging.info("---- Starting YOLO Unified (threaded) ----")
    logging.info("Logfile: %s", LOGFILE)
    try:
        run_unified(args)
    except Exception:
        logging.exception("Failed to start unified viewer")
        with open(CRASHLOG, "a") as f:
            f.write(datetime.utcnow().isoformat() + " CRASH on startup:\n")
            traceback.print_exc(file=f)


if __name__ == "__main__":
    main()
