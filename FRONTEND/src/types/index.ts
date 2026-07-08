export interface Profile {
  id: string;
  full_name: string;
  organization: string;
  role: string;
  avatar_url: string;
  created_at: string;
  updated_at: string;
}

export interface NotebookFile {
  id: string;
  user_id: string;
  phase: number;
  filename: string;
  content: NotebookContent;
  file_size: number;
  uploaded_at: string;
}

export interface NotebookContent {
  cells?: NotebookCell[];
  metadata?: Record<string, unknown>;
  nbformat?: number;
  nbformat_minor?: number;
}

export interface NotebookCell {
  cell_type: 'code' | 'markdown' | 'raw';
  source: string | string[];
  outputs?: CellOutput[];
  metadata?: Record<string, unknown>;
  execution_count?: number | null;
}

export interface CellOutput {
  output_type: 'stream' | 'display_data' | 'execute_result' | 'error';
  text?: string | string[];
  data?: Record<string, unknown>;
  traceback?: string[];
  ename?: string;
  evalue?: string;
  name?: string;
}

export interface DetectionHistory {
  id: string;
  user_id: string;
  input_type: 'image' | 'video';
  filename: string;
  ship_count: number;
  confidence_avg: number;
  processing_time_ms: number;
  result_json: DetectionResult;
  created_at: string;
}

export interface DetectionResult {
  detections: Detection[];
  image_url?: string;
  frames?: FrameResult[];
}

export interface Detection {
  bbox: [number, number, number, number];
  confidence: number;
  class: string;
}

export interface FrameResult {
  frame_index: number;
  timestamp: number;
  detections: Detection[];
  image_url?: string;
}

export interface PhaseInfo {
  phase: number;
  title: string;
  subtitle: string;
  description: string;
  input: string;
  output: string;
  color: string;
  icon: string;
}

export const PHASES: PhaseInfo[] = [
  {
    phase: 0,
    title: 'Phase 0',
    subtitle: 'Data Organisation & Pairing',
    description: 'Establish 1-to-1 correspondence between SAR and optical images. Generate synthetic SAR via 6-stage simulation pipeline. Rename sequentially and clean the ground truth CSV.',
    input: 'Raw optical images (~192K) + train_ship_segmentations.csv',
    output: 'Paired train_SAR/, train_OPTICAL/ folders + cleaned CSV',
    color: 'sky',
    icon: 'Database',
  },
  {
    phase: 1,
    title: 'Phase 1',
    subtitle: 'Preprocessing',
    description: 'Standardise both modalities. SAR: radiometric calibration, Lee speckle filter, min-max normalisation. Optical: brightness/contrast jitter. No geometric augmentation — YOLO labels tied to original orientation.',
    input: 'Paired raw SAR + optical images',
    output: 'Cleaned SAR tensors + augmented optical tensors (DataLoader on-the-fly)',
    color: 'teal',
    icon: 'Sliders',
  },
  {
    phase: 2,
    title: 'Phase 2',
    subtitle: 'Annotation & Label Generation',
    description: 'Decode RLE masks from CSV into YOLO bounding-box .txt files. One file per image; empty file for background images (essential for false-positive reduction).',
    input: 'Cleaned CSV with RLE masks + paired image index',
    output: 'YOLO .txt label files (one per image)',
    color: 'cyan',
    icon: 'Tag',
  },
  {
    phase: 3,
    title: 'Phase 3',
    subtitle: 'Dataset Loading',
    description: 'Build PyTorch Dataset serving (SAR, optical, labels) triplets. Stratified 5K ship + 5K background subset. 90/10 train/val split via random_split.',
    input: 'Paired images + YOLO labels + stratified index',
    output: 'train_loader and val_loader DataLoaders',
    color: 'blue',
    icon: 'Layers',
  },
  {
    phase: 4,
    title: 'Phase 4',
    subtitle: 'Dual-Stream Backbone',
    description: 'ResNet50 (1-ch SAR input) + CSPDarknet (3-ch optical) run in parallel. Both produce feature maps at P3/P4/P5 pyramid levels (stride 8/16/32).',
    input: 'SAR tensor [B,1,H,W] + Optical tensor [B,3,H,W]',
    output: 'Fs_P3/P4/P5 (SAR features) + Fo_P3/P4/P5 (Optical features)',
    color: 'indigo',
    icon: 'Cpu',
  },
  {
    phase: 5,
    title: 'Phase 5',
    subtitle: 'Multimodal Fusion (CrossModalAttention + FPN)',
    description: 'Concatenate SAR & optical features channel-wise at each pyramid level. CrossModalAttention computes cross-modal weights. FPN neck aggregates multi-scale features top-down.',
    input: 'Fs_P3/P4/P5 + Fo_P3/P4/P5',
    output: 'Fused FPN maps F_P3, F_P4, F_P5 [B, 256, H, W]',
    color: 'violet',
    icon: 'GitMerge',
  },
  {
    phase: 6,
    title: 'Phase 6',
    subtitle: 'Detection Head',
    description: 'Three parallel YOLO heads at P3/P4/P5. Each predicts x, y, w, h, objectness, class per anchor. NUM_ANCHORS=3, NUM_CLASSES=1 (ship). Output: 6 values per anchor per cell.',
    input: 'Fused FPN feature maps F_P3, F_P4, F_P5',
    output: 'Raw prediction tensors at 3 scales (decoded boxes + scores)',
    color: 'amber',
    icon: 'Crosshair',
  },
  {
    phase: 7,
    title: 'Phase 7',
    subtitle: 'Loss Computation & Training',
    description: 'L_total = L_cls (BCE) + L_bbox (CIoU) + L_obj (BCE). AdamW optimiser + CosineAnnealingLR. Checkpoint every 5 epochs; best model saved by val mAP.',
    input: 'Predicted boxes (Phase 6) + Ground truth YOLO labels',
    output: 'Trained model weights + per-epoch train/val loss CSV',
    color: 'orange',
    icon: 'TrendingUp',
  },
  {
    phase: 8,
    title: 'Phase 8',
    subtitle: 'Evaluation',
    description: 'Load best checkpoint. Run NMS (IoU=0.45). Compute mAP@0.5, Precision, Recall on 10% val split. TP if IoU ≥ 0.5. Precision-recall curve + confusion matrix.',
    input: 'Best model checkpoint + validation SAR/optical pairs',
    output: 'mAP@0.5, Precision, Recall, PR curve, confusion matrix',
    color: 'emerald',
    icon: 'BarChart2',
  },
];
