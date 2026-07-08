# MarineVision — Multimodal Maritime Object Detection Platform

**Team 22UG0005 | Panel 5 | Guide: Mr. Nagesh Koundinya Subbanna**

A full-stack web platform for multimodal maritime ship detection using SAR and optical satellite imagery. Combines a React/TypeScript frontend with a FastAPI Python backend that runs the trained PyTorch model (`best_model.pth`).

---

## Architecture

| Component | Technology |
|-----------|-----------|
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS |
| Backend | FastAPI (Python 3.10+) |
| Database | Supabase (PostgreSQL + Auth) |
| Model | PyTorch — ResNet50 + CSPDarknet + CrossModalAttention + FPN |

---

## Project Structure

```
project/
├── src/                        # React frontend
│   ├── pages/
│   │   ├── AuthPage.tsx        # Login / Register
│   │   ├── DashboardPage.tsx   # Overview + stats
│   │   ├── PipelinePage.tsx    # Phase overview (0–8)
│   │   ├── PhasePage.tsx       # Per-phase notebook viewer
│   │   ├── DetectionPage.tsx   # Ship detection (upload + result)
│   │   ├── AccountPage.tsx     # User profile + history
│   │   └── ArchitecturePage.tsx# System architecture diagram
│   ├── components/
│   │   ├── Layout.tsx
│   │   └── Sidebar.tsx
│   ├── hooks/useAuth.ts
│   ├── lib/supabase.ts
│   └── types/index.ts
├── backend/                    # Python FastAPI server
│   ├── main.py                 # API endpoints
│   ├── model.py                # PyTorch model definition
│   ├── requirements.txt
│   └── best_model.pth          # ← copy your trained model here
├── supabase/                   # Database migrations
└── .env                        # Supabase credentials
```

---

## Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+
- **Supabase** project (credentials in `.env`)
- Trained model file `best_model.pth`

---

## Setup & Running

### 1. Frontend

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend runs at `http://localhost:5173`.

### 2. Backend (model inference)

```bash
# Navigate to backend
cd backend

# Copy your trained model
cp /path/to/best_model.pth ./best_model.pth

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server
python main.py
```

The API runs at `http://localhost:8000`.

> **Without the backend running**, the Detection page will still work in demo mode with simulated results.

### 3. Environment Variables

The `.env` file must contain:

```
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key
VITE_BACKEND_URL=http://localhost:8000
```

---

## Features

### Authentication
- Email/password login and registration
- Secure session management via Supabase Auth

### Dashboard
- Live statistics (total scans, ships detected, confidence)
- Recent detection history
- Pipeline phase quick-access

### Ship Detection
- Upload **image** (PNG/JPG/TIFF) or **video** (MP4/AVI/MOV)
- Runs `best_model.pth` inference on the backend
- Displays annotated output with bounding boxes
- Saves result to detection history
- Video: samples frames, runs detection on each

### Pipeline Phases (0–8)
- Per-phase detail page with methodology description
- Upload Jupyter notebook (`.ipynb`) for each phase
- View code cells and outputs inline (including images)
- Delete uploaded notebooks
- Notebooks stored securely in Supabase per user

### System Architecture
- Visual architecture diagram
- Hardware configuration table
- Full vs actual training parameter comparison

### Account / History
- Edit profile (name, organisation, role)
- Full detection history with ship counts and confidence
- Delete individual history entries

---

## Pipeline Phases

| Phase | Name | Description |
|-------|------|-------------|
| 0 | Data Organisation | Pair SAR/optical images, clean CSV |
| 1 | Preprocessing | SAR: Lee filter + normalisation. Optical: jitter |
| 2 | Annotation | RLE → YOLO bounding box labels |
| 3 | Dataset Loading | Stratified DataLoader (5K ship + 5K background) |
| 4 | Dual-Stream Backbone | ResNet50 (SAR) + CSPDarknet (optical) |
| 5 | Multimodal Fusion | CrossModalAttention + FPN neck |
| 6 | Detection Head | YOLO heads at P3/P4/P5 |
| 7 | Training | AdamW + CosineAnnealingLR + CIoU loss |
| 8 | Evaluation | mAP@0.5, Precision, Recall, PR curve |

---

## Model Details

| Parameter | Value |
|-----------|-------|
| SAR Backbone | ResNet50 (modified 1-ch input) |
| Optical Backbone | CSPDarknet |
| Fusion | CrossModalAttention (channel-wise) |
| Neck | FPN (256 channels) |
| Head | Multi-scale YOLO at strides 8/16/32 |
| Classes | 1 (ship) |
| Loss | L_cls (BCE) + L_bbox (CIoU) + L_obj (BCE) |
| Optimizer | AdamW (wd=0.01) + CosineAnnealingLR |
| Confidence threshold | 0.50 |
| NMS IoU threshold | 0.45 |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| POST | `/detect` | Run ship detection (multipart form: `file`, `input_type`) |

### Detection Request

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@image.png" \
  -F "input_type=image"
```

### Detection Response

```json
{
  "detections": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.91, "class": "ship"}
  ],
  "ship_count": 3,
  "confidence_avg": 0.847,
  "processing_time_ms": 342,
  "annotated_image": "<base64 PNG>"
}
```

---

## Uploading Jupyter Notebooks

1. Navigate to any Phase page (e.g., Phase 0 → Phase 8)
2. Click **Upload .ipynb**
3. Select the corresponding notebook file:
   - `PHASE0_Data_Organisation_FIXED.ipynb`
   - `PHASE1_Preprocessing.ipynb`
   - `PHASE2_Annotation.ipynb`
   - `PHASE3_DatasetLoading.ipynb`
   - `PHASE4A_Methodology_Backbone.ipynb`
   - `PHASE5_Fusion_FPN.ipynb`
   - `PHASE6_DetectionHead.ipynb`
   - `PHASE7_Training_FIXED.ipynb`
   - `PHASE8_FIXED.ipynb`
4. All code cells and outputs render inline
5. Click **Delete** to remove and upload a new version

---

## Build for Production

```bash
# Frontend build
npm run build

# Backend can be deployed with:
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```
