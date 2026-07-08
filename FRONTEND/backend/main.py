"""
MarineVision Backend — FastAPI server for ship detection inference.

Fixed from original:
  1. Uses updated model.py with exact Phase-7 architecture
  2. decode_predictions returns normalised [0,1] coords — scaled to pixels here
  3. Single-image upload uses the image as BOTH optical and SAR inputs
     (production flow: user uploads one image; SAR is derived from it)
  4. Confidence threshold lowered to 0.05 for initial decode; NMS keeps ≥0.50
  5. Video path fixed (OpenCV VideoCapture used directly on temp file)
"""

import io
import os
import time
import base64
import tempfile

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import uvicorn

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw

from model import (
    MultimodalDetector,
    decode_predictions,
    ANCHORS, STRIDES,
    NUM_CLASSES,
)

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_SIZE      = 320          # must match Phase 7 training
DECODE_CONF = 0.35
NMS_CONF    = 0.85
NMS_IOU     = 0.20

ANCHORS_LIST = [ANCHORS['P3'], ANCHORS['P4'], ANCHORS['P5']]
STRIDES_LIST = [STRIDES['P3'], STRIDES['P4'], STRIDES['P5']]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
device     = torch.device("cpu")
model      = None

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MarineVision Detection API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}. Running in demo mode.")
        return

    try:
        model = MultimodalDetector(pretrained=False)

        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # Support various checkpoint formats
        if isinstance(checkpoint, dict):
            state = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("model_state")
                or checkpoint.get("state_dict")
                or checkpoint
            )
        else:
            state = checkpoint

        # Strip DataParallel 'module.' prefix if present
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model_keys = set(model.state_dict().keys())
        ckpt_keys  = set(state.keys())
        missing    = model_keys - ckpt_keys
        unexpected = ckpt_keys  - model_keys

        print(f"DEBUG checkpoint keys: {len(ckpt_keys)}, "
              f"model keys: {len(model_keys)}, "
              f"missing: {len(missing)}, "
              f"unexpected: {len(unexpected)}")

        if missing:
            print(f"  Missing (first 10):    {sorted(missing)[:10]}")
        if unexpected:
            print(f"  Unexpected (first 10): {sorted(unexpected)[:10]}")

        result = model.load_state_dict(state, strict=False)
        model.eval()

        if not missing and not unexpected:
            print("✓ Model loaded perfectly — all keys matched.")
        else:
            print(f"Model loaded with strict=False "
                  f"({len(missing)} missing / {len(unexpected)} unexpected).")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None


# ── Pre-processing ────────────────────────────────────────────────────────────
_opt_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def preprocess_optical(img: Image.Image) -> torch.Tensor:
    """RGB optical image → [1, 3, H, W]."""
    return _opt_transform(img.convert("RGB")).unsqueeze(0)


def preprocess_sar(img: Image.Image) -> torch.Tensor:
    """Greyscale SAR simulation: convert to L, min-max normalise → [1, 1, H, W]."""
    arr = np.array(img.convert("L"), dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-6:
        arr = (arr - mn) / (mx - mn)
    arr = cv2.resize(arr, (IMAGE_SIZE, IMAGE_SIZE))
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(opt_tensor: torch.Tensor,
                  sar_tensor: torch.Tensor,
                  orig_w: int,
                  orig_h: int) -> list:
    """
    Run model inference and return a list of detection dicts with PIXEL coords:
      [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class': 'ship'}, ...]
    """
    if model is None:
        return _demo_detections(orig_w, orig_h)

    with torch.no_grad():
        preds = model(sar_tensor, opt_tensor)   # (pred_P3, pred_P4, pred_P5)

    # decode_predictions returns normalised [0,1] coords, per image
    batched = decode_predictions(
        preds,
        anchors_list=ANCHORS_LIST,
        strides_list=STRIDES_LIST,
        device=device,
        conf_thresh=DECODE_CONF,
        nms_thresh=NMS_IOU,
        img_h=IMAGE_SIZE,
        img_w=IMAGE_SIZE,
    )

    raw_dets = batched[0]   # first (and only) image in batch

    # Debug output
    if raw_dets:
        confs = [d['confidence'] for d in raw_dets]
        print(f"DEBUG: {len(raw_dets)} dets after NMS  "
              f"min={min(confs):.3f} max={max(confs):.3f} "
              f"mean={sum(confs)/len(confs):.3f}")
    else:
        print("DEBUG: 0 detections after NMS")

    # Filter by NMS_CONF and convert normalised → pixel coords
    result = []
    for d in raw_dets:
        if d['confidence'] < NMS_CONF:
            continue
        result.append({
            "bbox": [
                float(d['x1']) * orig_w,
                float(d['y1']) * orig_h,
                float(d['x2']) * orig_w,
                float(d['y2']) * orig_h,
            ],
            "confidence": float(d['confidence']),
            "class": "ship",
        })

    # Sort strongest first
    raw_dets = sorted(raw_dets, key=lambda x: x['confidence'], reverse=True)

    result = []

    for d in raw_dets:

        if d['confidence'] < NMS_CONF:
            continue

        x1 = float(d['x1']) * orig_w
        y1 = float(d['y1']) * orig_h
        x2 = float(d['x2']) * orig_w
        y2 = float(d['y2']) * orig_h

        w = x2 - x1
        h = y2 - y1
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

# Reject unrealistic boxes
        if aspect_ratio > 12:
            continue

    # Remove tiny noisy boxes
        if w < 12 or h < 12:
            continue

    # Remove extremely large garbage boxes
        if w > orig_w * 0.5 or h > orig_h * 0.5:
           continue

        new_box = [x1, y1, x2, y2]

    # Strong duplicate suppression
        duplicate = False

        for existing in result:

            ex1, ey1, ex2, ey2 = existing["bbox"]

            # Existing center
            ecx = (ex1 + ex2) / 2
            ecy = (ey1 + ey2) / 2

    # New center
            ncx = (x1 + x2) / 2
            ncy = (y1 + y2) / 2

            center_dist = ((ecx - ncx) ** 2 + (ecy - ncy) ** 2) ** 0.5

    # Average box size
            avg_size = (
                ((ex2 - ex1) + (ey2 - ey1) + (x2 - x1) + (y2 - y1))
                / 4
            )

    # Same ship if centers too close
            if center_dist < avg_size * 1.2:
                duplicate = True
                break

        if duplicate:
            continue

        result.append({
            "bbox": new_box,
            "confidence": float(d['confidence']),
            "class": "ship",
       })

    # Keep only best detection
        if len(result) >= 5:
            break

    print(f"DEBUG: {len(result)} final detections")
    return result


def _demo_detections(orig_w: int, orig_h: int) -> list:
    """Return a plausible fake detection when the model is not loaded."""
    cx, cy = orig_w * 0.55, orig_h * 0.45
    bw, bh = orig_w * 0.18, orig_h * 0.12
    return [{
        "bbox": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
        "confidence": 0.82,
        "class": "ship",
    }]


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_boxes(img: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes directly in pixel coordinates."""
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        draw.rectangle([x1, y1, x2, y2], outline="#00bfff", width=3)
        label = f"ship {conf:.2f}"
        tw = len(label) * 7
        draw.rectangle([x1, max(0, y1 - 16), x1 + tw, y1], fill="#00bfff")
        draw.text((x1 + 2, max(0, y1 - 15)), label, fill="white")
    return img


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_model()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "loaded" if model is not None else "not found",
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    input_type: str = Form("image"),
):
    start = time.time()
    data  = await file.read()

    # ── Image ──────────────────────────────────────────────────────────────
    if input_type == "image":
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        orig_w, orig_h = img.size
        opt_tensor = preprocess_optical(img)
        sar_tensor = preprocess_sar(img)

        detections = run_inference(opt_tensor, sar_tensor, orig_w, orig_h)

        annotated = img.copy()
        annotated = draw_boxes(annotated, detections)

        conf_avg = (float(np.mean([d["confidence"] for d in detections]))
                    if detections else 0.0)
        elapsed  = int((time.time() - start) * 1000)

        return JSONResponse({
            "detections":       detections,
            "ship_count":       len(detections),
            "confidence_avg":   round(conf_avg, 4),
            "processing_time_ms": elapsed,
            "annotated_image":  pil_to_base64(annotated),
        })

    # ── Video ──────────────────────────────────────────────────────────────
    elif input_type == "video":
        suffix = os.path.splitext(file.filename or "vid.mp4")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail="Cannot decode video")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        sample_count = min(16, max(1, total_frames))
        step         = max(1, total_frames // sample_count)

        frames_result = []
        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0 and processed < sample_count:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                orig_w, orig_h = img.size
                opt_tensor = preprocess_optical(img)
                sar_tensor = preprocess_sar(img)
                dets = run_inference(opt_tensor, sar_tensor, orig_w, orig_h)
                ann  = draw_boxes(img.copy(), dets)
                frames_result.append({
                    "frame_index": frame_idx,
                    "timestamp":   round(frame_idx / fps, 2),
                    "detections":  dets,
                    "image":       pil_to_base64(ann),
                })
                processed += 1
            frame_idx += 1

        cap.release()
        os.unlink(tmp_path)

        all_dets = [d for fr in frames_result for d in fr["detections"]]
        conf_avg = (float(np.mean([d["confidence"] for d in all_dets]))
                    if all_dets else 0.0)
        elapsed  = int((time.time() - start) * 1000)

        return JSONResponse({
            "detections":         all_dets,
            "ship_count":         len(all_dets),
            "confidence_avg":     round(conf_avg, 4),
            "processing_time_ms": elapsed,
            "frames":             frames_result,
        })

    raise HTTPException(status_code=400, detail="input_type must be 'image' or 'video'")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)