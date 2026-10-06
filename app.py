"""
SixRay v3 - FastAPI Web Server
================================
Serves the HTML frontend and exposes two API endpoints:
  POST /api/detect  - Run YOLO detection on an uploaded image
  GET  /api/metrics - Return training metrics from results.csv
"""

import base64
import csv
import io
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS     = PROJECT_DIR / "runs" / "detect" / "sixray_model" / "weights" / "best.pt"
RESULTS_CSV = PROJECT_DIR / "runs" / "detect" / "sixray_model" / "results.csv"
STATIC_DIR  = PROJECT_DIR / "static"

CONFIDENCE  = 0.25
BOX_COLOR   = (0, 255, 0)   # BGR green

# ── Startup ────────────────────────────────────────────────────────────────
app = FastAPI(title="SixRay v3 API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
if not WEIGHTS.exists():
    raise RuntimeError(f"Model weights not found at {WEIGHTS}. Train the model first.")

print(f"Loading model from: {WEIGHTS}")
model      = YOLO(str(WEIGHTS))
CLASS_NAMES = model.names   # {0: 'Gun', 1: 'Knife', ...}
print(f"Classes: {list(CLASS_NAMES.values())}")


# ── Helpers ────────────────────────────────────────────────────────────────
def _encode_image(frame: np.ndarray) -> str:
    """Encode a cv2 frame to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buf).decode()


def _draw_boxes(frame: np.ndarray, results) -> tuple[np.ndarray, list]:
    """Draw bounding boxes on frame; return annotated frame and detections list."""
    detections = []
    font       = cv2.FONT_HERSHEY_SIMPLEX

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = f"{CLASS_NAMES[cls_id]} {conf:.0%}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
            (tw, th), _ = cv2.getTextSize(label, font, 0.65, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), BOX_COLOR, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4), font, 0.65, (0, 0, 0), 2)

            detections.append({
                "class": CLASS_NAMES[cls_id],
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2],
            })

    return frame, detections


# ── API Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.25):
    """
    Accept an image upload, run YOLO detection, return:
      - annotated_image : base64-encoded JPEG
      - detections      : list of {class, confidence, bbox}
      - image_size      : {width, height}
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    arr   = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    h, w = frame.shape[:2]

    # Convert to grayscale then back (matches detect_image.py behaviour)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    results        = model(frame, conf=conf, verbose=False)
    frame, detects = _draw_boxes(frame, results)

    return JSONResponse({
        "annotated_image": _encode_image(frame),
        "detections":      detects,
        "image_size":      {"width": w, "height": h},
        "total_detected":  len(detects),
    })


@app.get("/api/metrics")
def get_metrics():
    """
    Parse results.csv from the training run and return epoch-level metrics.
    """
    if not RESULTS_CSV.exists():
        raise HTTPException(status_code=404, detail="results.csv not found. Train the model first.")

    rows = []
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items()}
            try:
                rows.append({
                    "epoch":        int(float(cleaned.get("epoch", 0))),
                    "map50":        float(cleaned.get("metrics/mAP50(B)", 0)),
                    "map50_95":     float(cleaned.get("metrics/mAP50-95(B)", 0)),
                    "precision":    float(cleaned.get("metrics/precision(B)", 0)),
                    "recall":       float(cleaned.get("metrics/recall(B)", 0)),
                    "train_box_loss": float(cleaned.get("train/box_loss", 0)),
                    "val_box_loss":   float(cleaned.get("val/box_loss", 0)),
                })
            except (ValueError, KeyError):
                continue

    return JSONResponse({
        "epochs":       rows,
        "best_map50":   max((r["map50"] for r in rows), default=0),
        "best_map5095": max((r["map50_95"] for r in rows), default=0),
        "final_precision": rows[-1]["precision"] if rows else 0,
        "final_recall":    rows[-1]["recall"] if rows else 0,
    })


@app.get("/api/classes")
def get_classes():
    return JSONResponse({"classes": list(CLASS_NAMES.values())})


# ── Static Files (serve frontend) ──────────────────────────────────────────
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 50)
    print("  SixRay v3 App Running Locally")
    print("  Access web UI at: http://127.0.0.1:8000")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)

