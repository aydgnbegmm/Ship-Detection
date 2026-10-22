"""
Maritime Ship Detection Inference Script
Developed within the scope of SAYZEK-ATP (Defense Industry AI Talent Clustering)
Academic Thesis: Deep Learning-Based Computer Vision Techniques for Autonomous UAV Landing on Maritime Vessels

Author: Begüm Aydoğan Çiçin
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Maritime Ship Detection & Tracking Inference")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to trained YOLOv8 model weights")
    parser.add_argument("--source", type=str, default="data/test_samples", help="Source directory, image, video, or camera index (0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution (pixels)")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="", help="CUDA device, i.e. 0 or cpu")
    parser.add_argument("--show", action="store_true", help="Display visual results in real time")
    parser.add_argument("--save", action="store_true", help="Save annotated images or videos to output directory")
    parser.add_argument("--output", type=str, default="output", help="Directory to save output files")
    return parser.parse_args()


def run_inference():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading model from: {args.weights}")
    model = YOLO(args.weights)

    # Determine source type
    is_cam = args.source.isdigit()
    source = int(args.source) if is_cam else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and not Path(args.source).is_dir():
        print(f"[!] Error: Could not open media source: {args.source}")
        return

    frame_count = 0
    start_time = time.time()

    print("[*] Starting maritime detection inference loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.time()

        # Run YOLO inference
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False
        )

        infer_time_ms = (time.time() - t0) * 1000
        annotated_frame = results[0].plot()

        # Overlay telemetry / inference metrics
        fps = 1000.0 / infer_time_ms if infer_time_ms > 0 else 0.0
        cv2.putText(
            annotated_frame,
            f"Inference: {infer_time_ms:.1f}ms ({fps:.1f} FPS)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        detections = results[0].boxes
        cv2.putText(
            annotated_frame,
            f"Vessels Detected: {len(detections)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if args.show:
            cv2.imshow("Maritime Ship Detection (SAYZEK-ATP)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"[*] Completed processing {frame_count} frames in {elapsed:.2f} seconds ({frame_count/max(elapsed, 0.001):.1f} avg FPS).")


if __name__ == "__main__":
    run_inference()
