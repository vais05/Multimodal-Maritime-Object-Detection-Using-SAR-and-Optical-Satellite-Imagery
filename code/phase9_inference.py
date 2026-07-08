"""
Phase 9 — Inference on new SAR + Optical images using best_model.onnx
"""

import numpy as np
import cv2
import onnxruntime as ort

# ── Config ────────────────────────────────────────────────────────────────────
ONNX_PATH = r'C:\Users\VaishnaviM\Downloads\airbus-ship-detection\dataset\checkpoints\best_model.onnx'

IMAGE_SIZE = 320
CONF_THRESH = 0.93
NMS_THRESH = 0.5

NUM_ANCHORS = 3
NUM_CLASSES = 1

ANCHORS = {
    'P3': [(6,4),   (10,6),  (14,8)],
    'P4': [(20,12), (28,18), (36,24)],
    'P5': [(50,30), (70,45), (90,60)],
}

STRIDES = {
    'P3': 8,
    'P4': 16,
    'P5': 32
}

# ── Load ONNX Model ───────────────────────────────────────────────────────────
session = ort.InferenceSession(
    ONNX_PATH,
    providers=['CPUExecutionProvider']
)

print("OK ONNX model loaded")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_sar(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"SAR image not found: {path}")

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0

    return img[np.newaxis, np.newaxis, :, :]


def preprocess_optical(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Optical image not found: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0

    return img.transpose(2, 0, 1)[np.newaxis, :, :, :]


# ── Decode YOLO Head ──────────────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode_head(pred, anchors, stride):

    B, _, H, W = pred.shape

    pred = pred.reshape(
        B,
        NUM_ANCHORS,
        5 + NUM_CLASSES,
        H,
        W
    )

    pred = pred.transpose(0, 1, 3, 4, 2)

    boxes = []
    scores = []

    for a_idx, (aw, ah) in enumerate(anchors):

        for cy in range(H):
            for cx in range(W):

                v = pred[0, a_idx, cy, cx]

                tx, ty, tw, th, obj = v[0], v[1], v[2], v[3], v[4]
                cls_conf = v[5]

                obj_score = sigmoid(obj)
                cls_score = sigmoid(cls_conf)

                conf = obj_score * cls_score

                if conf < CONF_THRESH:
                    continue

                x_c = (sigmoid(tx) + cx) * stride
                y_c = (sigmoid(ty) + cy) * stride

                w = aw * np.exp(tw)
                h = ah * np.exp(th)

                x1 = max(0, x_c - w / 2)
                y1 = max(0, y_c - h / 2)
                x2 = min(IMAGE_SIZE, x_c + w / 2)
                y2 = min(IMAGE_SIZE, y_c + h / 2)

                bw = x2 - x1
                bh = y2 - y1

                # Remove tiny noise boxes
                if bw < 8 or bh < 8:
                    continue

                # Remove huge false detections
                if bw > 120 or bh > 120:
                    continue

                boxes.append([x1, y1, x2, y2])
                scores.append(float(conf))

    return boxes, scores


# ── Strong NMS ────────────────────────────────────────────────────────────────
def nms(boxes, scores):

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:

        i = order[0]

        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h

        overlap = inter / (areas[order[1:]] + 1e-6)

        inds = np.where(overlap <= NMS_THRESH)[0]

        order = order[inds + 1]

    return keep


# ── Full Detection Pipeline ───────────────────────────────────────────────────
def detect(sar_path, optical_path):

    sar = preprocess_sar(sar_path)
    optical = preprocess_optical(optical_path)

    p3, p4, p5 = session.run(
        ['pred_P3', 'pred_P4', 'pred_P5'],
        {
            'sar': sar,
            'optical': optical
        }
    )

    all_boxes = []
    all_scores = []

    for pred, key in [
        (p3, 'P3'),
        (p4, 'P4'),
        (p5, 'P5')
    ]:

        boxes, scores = decode_head(
            pred,
            ANCHORS[key],
            STRIDES[key]
        )

        all_boxes.extend(boxes)
        all_scores.extend(scores)

    print("Raw detections:", len(all_boxes))

    keep = nms(all_boxes, all_scores)

    final_boxes = [all_boxes[i] for i in keep]
    final_scores = [all_scores[i] for i in keep]

    # Keep ONLY highest confidence detection
    if len(final_scores) > 0:

        best_idx = np.argmax(final_scores)

        final_boxes = [final_boxes[best_idx]]
        final_scores = [final_scores[best_idx]]

    print(f"OK {len(final_boxes)} ship(s) detected")

    # ── Draw Detection ────────────────────────────────────────────────────────
    display_img = cv2.imread(optical_path)

    for i, (box, score) in enumerate(zip(final_boxes, final_scores)):

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            display_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"Ship {score:.2f}"

        cv2.putText(
            display_img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        print(
            f"Ship {i+1}: "
            f"conf={score:.3f} "
            f"box=[{x1},{y1},{x2},{y2}]"
        )

    cv2.imshow("Ship Detection", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final_boxes, final_scores


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    SAR_IMG = r'C:\Users\VaishnaviM\Downloads\airbus-ship-detection\dataset\preprocessed_test_SAR\00156_sar.png'

    OPTICAL_IMG = r'C:\Users\VaishnaviM\Downloads\airbus-ship-detection\dataset\preprocessed_test_OPTICAL\00156_opt.png'

    detect(SAR_IMG, OPTICAL_IMG)