#!/usr/bin/env python3
"""
Headless capture: grab N frames from camera, run YOLO (optionally MiDaS), write detections CSV, then exit.

Usage:
  source .venv/bin/activate
  python3 examples/vision/webcam_yolo_capture.py --frames 200 --device cpu --out detections_batch.csv --midas

Notes:
- Uses ultralytics YOLO (yolov8n by default). First run will download weights.
- --midas enables MiDaS relative depth (slow on CPU).
"""
import argparse
import csv
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def load_midas(device, model_name="MiDaS_small"):
    model = torch.hub.load("intel-isl/MiDaS", model_name)
    model.to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform if model_name == "MiDaS_small" else transforms.default_transform
    return model, transform

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="yolov8n")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--frames", type=int, default=200, help="number of frames to capture")
    ap.add_argument("--out", default="detections_batch.csv", help="output CSV")
    ap.add_argument("--midas", action="store_true", help="run MiDaS for relative depth")
    ap.add_argument("--conf", type=float, default=0.25, help="min confidence")
    args = ap.parse_args()

    device = args.device
    print("Device:", device, "Model:", args.model, "Camera:", args.camera, "Frames:", args.frames, "MiDaS:", args.midas)
    model = YOLO(args.model)

    midas_model = None
    midas_transform = None
    if args.midas:
        print("Loading MiDaS (this may download weights)...")
        midas_model, midas_transform = load_midas(device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("ERROR: camera not available (index {})".format(args.camera))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "label", "conf", "x1", "y1", "x2", "y2", "cx", "cy", "depth_rel"])
        for i in range(args.frames):
            ret, frame = cap.read()
            if not ret:
                print("frame grab failed at", i)
                break

            depth_map = None
            if midas_model is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_t = midas_transform(img_rgb).to(device)
                with torch.no_grad():
                    pred = midas_model(input_t)
                    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=img_rgb.shape[:2],
                                                           mode="bilinear", align_corners=False).squeeze()
                    depth_map = pred.cpu().numpy()

            results = model(frame, imgsz=640, device=device, conf=args.conf)
            ts = time.time()
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
                    depth_val = ""
                    if depth_map is not None:
                        vy1, vy2 = max(0, cy-2), min(depth_map.shape[0], cy+3)
                        vx1, vx2 = max(0, cx-2), min(depth_map.shape[1], cx+3)
                        window = depth_map[vy1:vy2, vx1:vx2]
                        if window.size:
                            depth_val = float(np.median(window))
                    label = res.names[int(cls)] if hasattr(res, "names") else str(cls)
                    writer.writerow([ts, label, f"{conf:.4f}", x1, y1, x2, y2, cx, cy, depth_val])
            # optional: small sleep to reduce CPU usage
            # time.sleep(0.01)

    cap.release()
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
