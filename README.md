# Multimodal Maritime Object Detection Using SAR and Optical Satellite Imagery


## 1. What This Project Does

Ships need to be detected in satellite images for coastal security, illegal-fishing detection, port monitoring, and search-and-rescue. The problem: **optical satellite images** look great in clear weather but fail under cloud, fog, rain, or at night. **SAR (Synthetic Aperture Radar) images** work in any weather or lighting, but are noisy (speckle) and visually hard to interpret.

This project fuses **both modalities** in a single deep learning model so the system gets the best of both: the rich texture/color detail of optical imagery and the all-weather reliability of SAR. The result is a **dual-stream YOLO-style detector** that takes a paired SAR + optical image and outputs bounding boxes around ships.

Because the entire project was trained on a **CPU-only laptop** (no GPU), a large part of the engineering effort was making a normally GPU-hungry architecture trainable on modest hardware (Intel i5-12450H, 16GB RAM) without losing the core design.

---

## 2. Architecture Overview

```
                        ┌──────────────────┐        ┌──────────────────┐
                        │   SAR image      │        │  Optical image   │
                        │ [B,1,320,320]    │        │ [B,3,320,320]    │
                        └────────┬─────────┘        └────────┬─────────┘
                                 │                             │
                        ┌────────▼─────────┐        ┌─────────▼────────┐
                        │  SAR Backbone     │        │ Optical Backbone │
                        │  ResNet50         │        │ CSPDarknet       │
                        │ (pretrained,      │        │ (pretrained,     │
                        │  1-channel input) │        │  YOLO-style)     │
                        └────────┬─────────┘        └─────────┬────────┘
                          Fs_P3/P4/P5                    Fo_P3/P4/P5
                                 │                             │
                                 └───────────┬─────────────────┘
                                             ▼
                              ┌───────────────────────────┐
                              │  CrossModalAttention      │
                              │  fusion (per pyramid level)│
                              │  concat → attention weights│
                              └────────────┬───────────────┘
                                           ▼
                              ┌───────────────────────────┐
                              │  FPN Neck (256 channels)  │
                              │  top-down aggregation      │
                              └────────────┬───────────────┘
                                    F_P3 / F_P4 / F_P5
                                           │
                     ┌─────────────────────┼─────────────────────┐
                     ▼                     ▼                     ▼
              ┌─────────────┐      ┌─────────────┐        ┌─────────────┐
              │ Head P3     │      │ Head P4     │        │ Head P5     │
              │ stride 8    │      │ stride 16   │        │ stride 32   │
              │ small ships │      │ medium ships│        │ large ships │
              └──────┬──────┘      └──────┬──────┘        └──────┬──────┘
                     └─────────────────────┼─────────────────────┘
                                           ▼
                              ┌───────────────────────────┐
                              │  NMS (IoU 0.45, conf 0.5) │
                              └────────────┬───────────────┘
                                           ▼
                          Bounding boxes + confidence scores + annotated images
```

| Component | Choice | Why |
|---|---|---|
| SAR backbone | ResNet50 (ImageNet pretrained, first conv modified to 1-channel) | Deep hierarchical features, robust to speckle-heavy input |
| Optical backbone | CSPDarknet (YOLO-style pretrained) | Standard, efficient RGB feature extractor used across YOLO family |
| Fusion | CrossModalAttention | Learns *how much* to trust each modality per region, instead of naive concatenation |
| Neck | FPN, 256 channels | Combines shallow (spatial) + deep (semantic) features for multi-scale detection |
| Heads | 3× YOLO-style heads (P3/P4/P5) | Detect small, medium, and large ships separately |
| Loss | `L_total = L_cls + L_bbox + L_obj` | Classification + CIoU box regression + objectness |
| Optimizer | AdamW + CosineAnnealingLR | Stable convergence, smooth LR decay |

---

## 3. The 9-Phase Pipeline

The whole system is broken into 9 sequential phases, each with a clear input → output contract. This mirrors the LLD flowchart (Figure 4.2) in the project report.

### Phase 0 — Data Organisation & Pairing
- Pairs each optical image with a corresponding SAR image (~192,556 optical images from the Airbus Ship Detection Dataset).
- Since a real large-scale paired SAR dataset wasn't available, SAR images are **synthetically generated** from optical images via a 6-stage optical→SAR simulation (radiometric profile → speckle noise → edge sharpening → texture suppression → tonal shift → contrast compression), saved as 16-bit PNGs.
- Files renamed sequentially (`0001_opt.png` / `0001_sar.png`, …).
- **Critical bug fixed here:** the annotation CSV (`train_ship_segmentations.csv`) had ~3,841 stale rows using old hex-format image IDs left over from an interrupted rename operation. These rows are filtered out before Phase 2, otherwise labels get mismatched to the wrong images.
- **Output:** `train_SAR/`, `train_OPTICAL/`, `test_SAR/`, `test_OPTICAL/`, and a cleaned CSV.

### Phase 1 — Preprocessing (`preprocess.py`)
- **SAR side:** radiometric calibration → Lee filter (5×5) for speckle noise removal (preserves ship edges, unlike a plain blur) → min-max normalization to [0,1] → bilinear resize to 320×320.
- **Optical side:** resize → brightness/contrast jitter (±20%) → mild random crop-and-resize.
- **No rotation or flipping is applied to either modality.** This is deliberate: YOLO labels are tied to the original image orientation, and a geometric transform without recomputing the label coordinates would silently corrupt the ground truth.
- All of this happens on-the-fly inside the DataLoader — nothing augmented is written to disk.

### Phase 2 — Annotation & Label Generation (`generate_labels.py`)
- Ship masks in the dataset are stored as **Run-Length Encoded (RLE)** strings, not bounding boxes.
- Steps: decode RLE → reshape into a 2D binary mask → `cv2.boundingRect()` on each connected region → normalize to YOLO format:

```
x_center = (x + w/2) / img_w
y_center = (y + h/2) / img_h
w_norm   = w / img_w
h_norm   = h / img_h
```

- One `.txt` label file is written per image (shared by SAR and optical since they depict the same scene). Images with no ships get an **empty** label file — these are kept deliberately as background/negative training samples.

### Phase 3 — Dataset Loading (`dataset.py`)
- A custom PyTorch `Dataset` + `DataLoader` serves `(sar_tensor, optical_tensor, labels)` triplets.
- **Class imbalance fix:** the raw dataset is ~78% background / 22% ship images. A **stratified subset** of 10,000 pairs (5,000 ship + 5,000 background) is drawn so the model doesn't collapse into "always predict background."
- 90/10 train/validation split via `random_split`.
- `pin_memory=True` for faster CPU→tensor transfer; `num_workers=2` (Windows caps this reliably).

### Phase 4 — Dual-Stream Feature Extraction (`model.py`)
- SAR tensor `[B,1,H,W]` → ResNet50 → `Fs_P3, Fs_P4, Fs_P5`
- Optical tensor `[B,3,H,W]` → CSPDarknet → `Fo_P3, Fo_P4, Fo_P5`
- Backbones are **frozen** initially (in the full design, only for the first few epochs, then unfrozen for fine-tuning — though in the actual CPU run, they were kept frozen for **all** 20 epochs purely to save compute).

### Phase 5 — Multimodal Fusion + FPN (`model.py`)
- SAR and optical feature maps are concatenated channel-wise at each pyramid level, then passed through **CrossModalAttention** to compute attention weights — this is what lets the network say "trust optical here, trust SAR there" instead of blending blindly.
- Fused maps go through an **FPN** (top-down pathway: upsample P5 → merge with P4 → merge with P3), producing `F_P3, F_P4, F_P5`, each with 256 channels.

### Phase 6 — Detection Head (`model.py`)
- Three YOLO-style heads, one per scale:

| Head | Stride | Grid (at 320×320) | Detects |
|---|---|---|---|
| P3 | 8 | 40×40 | Small ships |
| P4 | 16 | 20×20 | Medium ships |
| P5 | 32 | 10×10 | Large ships |

- Each head predicts, per anchor per grid cell: `x, y, w, h, objectness, class_confidence` → output shape `[B, 3×(5+1), H/stride, W/stride]`.
- 3 anchors per scale, 1 class (ship only).

### Phase 7 — Training (`train.py`)
- Loss: `L_total = L_cls (BCE) + L_bbox (CIoU) + L_obj (BCE)`
- Optimizer: **AdamW** (lr=0.001, weight_decay=0.01)
- Scheduler: **CosineAnnealingLR**
- Checkpoints saved periodically; **best checkpoint chosen by highest validation mAP**, not just lowest loss.
- CPU-only adaptations used in this run: image size 320×320 (not the designed 640×640), batch size 4, gradient accumulation of 2 steps (simulates an effective batch of 8), backbone frozen throughout, mixed precision disabled (has no effect on CPU anyway), `torch.set_num_threads(8)` to use all CPU cores, 20 epochs instead of the architecturally-planned 50.

### Phase 8 — Evaluation (`evaluate.py`)
- Load the best checkpoint → run inference on the held-out 10% validation split → apply NMS (IoU threshold 0.45) → compute Precision, Recall, IoU, AP, mAP@0.5.
- **Final results achieved:**

| Metric | Value |
|---|---|
| Precision | 89.00% |
| Recall | 93.56% |
| F1-Score | 91.22% |
| mAP@0.5 | 91.71% |
| True Positives | 947 |
| False Positives | 117 |
| False Negatives | 65 |

- mAP rose from a low starting value up to **91.71% at epoch 19** (best checkpoint), with a clear inflection around epoch 10 when backbone fine-tuning effects showed up.
- Scale-wise: P4 (medium ships) performed best since most Airbus ships fall in that size range; P3 (small ships) had the weakest recall due to low input resolution; P5 (large ships) had very high precision since large vessels have a strong, unambiguous SAR signature.

### Phase 9 — Inference (`inference.py`)
- Takes a **new, unseen** SAR + optical pair → preprocesses identically to Phase 1 (no augmentation) → forward pass → decodes boxes → confidence threshold 0.5 → NMS → draws boxes on both the optical and SAR image → saves annotated outputs with confidence scores.
- No ground truth, no loss, no weight updates — pure prediction.
- The trained PyTorch model (`best_model.pth`) was additionally converted to **ONNX** for lightweight deployment via ONNX Runtime.

---

## 4. Code File Map

| File | Responsibility |
|---|---|
| `preprocess.py` | SAR/optical normalization, Lee filtering, resizing, augmentation |
| `generate_labels.py` | RLE mask decoding → YOLO `.txt` label generation |
| `dataset.py` | Custom `Dataset`/`DataLoader`, stratified sampling, train/val split |
| `model.py` | ResNet50 + CSPDarknet backbones, CrossModalAttention fusion, FPN, YOLO heads |
| `train.py` | Training loop, loss computation, optimizer/scheduler, checkpointing |
| `evaluate.py` | mAP/Precision/Recall/IoU computation, PR curve, confusion matrix |
| `inference.py` | Loads best checkpoint (or ONNX model), runs detection on new image pairs, draws boxes |

---

## 5. Requirements

**Software:** Python 3.x, PyTorch, Torchvision, OpenCV, NumPy, Pandas, Matplotlib, PIL, Scikit-learn, Jupyter Notebook / VS Code, Windows 11.

**Hardware used for this submission:** Lenovo IdeaPad Slim 5 14IAH8, Intel Core i5-12450H (8 cores), 16GB DDR5 RAM, integrated Intel UHD graphics (**no CUDA/GPU acceleration**).

**Recommended for full-scale reproduction** (640×640 images, batch size 8, full 192K dataset, 50 epochs): NVIDIA RTX 3060 or better (8GB+ VRAM), 32GB system RAM, ~6–12 hours training time. No architectural code changes needed — only `IMAGE_SIZE`, `BATCH_SIZE`, `NUM_EPOCHS`, and the training subset size need updating, and mixed precision / backbone unfreezing-after-epoch-10 should be re-enabled.

---

## 6. Key Design Trade-offs (CPU Constraints)

| Parameter | Designed value | Used value | Reason |
|---|---|---|---|
| Image size | 640×640 | 320×320 | ~75% less memory; still valid since it's a multiple of 32 |
| Batch size | 8 | 4 | Prevents RAM overflow |
| Training samples | ~192,556 | 10,000 (stratified) | Full set infeasible on CPU |
| Epochs | 50 | 20 | ~45–90 min/epoch on CPU |
| Mixed precision | Enabled | Disabled | No effect without CUDA |
| Backbone freezing | First few epochs only | All epochs | Saves compute |

These are documented as **hardware-driven adaptations**, not changes to the underlying architecture — the same code, with the constants above changed, is designed to scale to full GPU training without modification.

---

## 7. Limitations & Honest Caveats

- 320×320 resolution loses fine detail for very small/distant ships — a known source of false negatives.
- Only a 10,000-image stratified subset of the full ~192K dataset was used for training due to CPU constraints.
- SAR images are **simulated** from optical images (6-stage synthetic pipeline), not real satellite SAR captures — there is a domain gap between simulated speckle/backscatter and real-world SAR, even though the fusion strategy and convergence behavior held up well.
- Single-class detection only (ship vs. background); no vessel-type classification.

---

## 8. Future Scope (from the report)

- Train on real-world SAR datasets (not simulated) for better generalization.
- Full GPU-scale training at 640×640 with the full dataset.
- Multi-class detection (cargo ships, fishing boats, tankers, naval vessels, passenger ships).
- Transformer-based backbones/heads (ViT, DETR, Swin).
- Real-time / edge deployment, and fusion with AIS (Automatic Identification System) data for dark-ship detection and anomaly/threat analysis.
- Temporal/video-based tracking across sequential SAR frames.

---

*This README was generated from the project report (`005.pdf`) and methodology document (`methodology_updated.pdf`) for Team 22UG0005's B.Tech final-year project.*
