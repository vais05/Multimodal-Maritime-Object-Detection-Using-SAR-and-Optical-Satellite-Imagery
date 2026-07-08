"""
MarineVision Model — rebuilt to exactly match best_model.pth checkpoint keys.

Checkpoint top-level modules:
  sar_backbone  — ResNet50 (1-ch): stem + layer1/2/3/4
  opt_backbone  — CSPDarknet: stem + stage1/2/3/4  (conv stored as .conv.0/.conv.1)
  fusion        — cma3/cma4/cma5 (CrossModalAttention) + fpn
  det_head      — head3/head4/head5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Shared conv-bn-silu building block ──────────────────────────────────────
# NOTE: checkpoint stores weights as  .conv.0.weight  (Conv2d)
#                                      .conv.1.*       (BatchNorm2d)
# So we wrap them in a sub-module called "conv".

class CBS(nn.Module):
    """Conv → BN → SiLU.  Weights live at  self.conv[0]  and  self.conv[1]."""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x))


# ─── SAR Backbone (ResNet50, 1-channel) ──────────────────────────────────────
# Checkpoint keys: sar_backbone.stem.0  (Conv2d weight)
#                  sar_backbone.stem.1  (BN)
#                  sar_backbone.layer1/2/3/4  — standard Bottleneck blocks

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch,  mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_resnet_layer(in_ch, mid_ch, blocks, stride=1):
    out_ch = mid_ch * Bottleneck.expansion
    downsample = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    layers = [Bottleneck(in_ch, mid_ch, stride=stride, downsample=downsample)]
    for _ in range(1, blocks):
        layers.append(Bottleneck(out_ch, mid_ch))
    return nn.Sequential(*layers)


class SARBackbone(nn.Module):
    """ResNet50 with 1-channel input.
    stem:   Conv2d(1→64, 7×7, s2) + BN   → stored as stem.0 / stem.1
    layer1: 3 × Bottleneck(64→256)
    layer2: 4 × Bottleneck(256→512,  s2)
    layer3: 6 × Bottleneck(512→1024, s2)
    layer4: 3 × Bottleneck(1024→2048,s2)
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),   # stem.0
            nn.BatchNorm2d(64),                                       # stem.1
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = _make_resnet_layer(64,   64,  3, stride=1)   # out 256
        self.layer2 = _make_resnet_layer(256,  128, 4, stride=2)   # out 512
        self.layer3 = _make_resnet_layer(512,  256, 6, stride=2)   # out 1024
        self.layer4 = _make_resnet_layer(1024, 512, 3, stride=2)   # out 2048

    def forward(self, x):
        x  = self.stem(x)
        x  = F.relu(x)
        x  = self.maxpool(x)
        x  = self.layer1(x)
        p3 = self.layer2(x)   # [B, 512,  H/8,  W/8]
        p4 = self.layer3(p3)  # [B, 1024, H/16, W/16]
        p5 = self.layer4(p4)  # [B, 2048, H/32, W/32]
        return p3, p4, p5


# ─── Optical Backbone (CSPDarknet) ───────────────────────────────────────────
# Checkpoint keys use  .conv.0  /  .conv.1  for every conv-bn pair.
# CSPBlock bottleneck units are named  bns.N.cv1  /  bns.N.cv2

class CSPBottleneck(nn.Module):
    """One residual unit inside CSPBlock: cv1 → cv2."""
    def __init__(self, ch):
        super().__init__()
        self.cv1 = CBS(ch, ch, 3, 1, 1)
        self.cv2 = CBS(ch, ch, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class CSPBlock(nn.Module):
    """CSP block: cv1 + bns (bottlenecks) + cv2 (skip) → cat → cv3."""
    def __init__(self, in_ch, out_ch, n=1):
        super().__init__()
        h = out_ch // 2
        self.cv1 = CBS(in_ch, h)
        self.bns = nn.ModuleList([CSPBottleneck(h) for _ in range(n)])
        self.cv2 = CBS(in_ch, h)
        self.cv3 = CBS(2 * h, out_ch)

    def forward(self, x):
        a = x
        for bn in self.bns:
            a = bn(a)
        a = self.cv1(a)      # main branch after bottlenecks
        # Wait — let's re-check the key order:
        # cv1 is applied first, then bns, then cv2 is the skip on raw x
        # Reorder to match typical CSP:
        #   cv1(x) → bottlenecks → one branch
        #   cv2(x)               → skip branch
        # But that would put cv1 before bns.  Let's trace keys again:
        # opt_backbone.stage1.1.cv1 / .bns.N / .cv2 / .cv3
        # Standard CSP: main = bns(cv1(x)), skip = cv2(x), out = cv3(cat(main,skip))
        b = self.cv2(x)
        return self.cv3(torch.cat([a, b], dim=1))

    def forward(self, x):
        main = self.cv1(x)
        for bn in self.bns:
            main = bn(main)
        skip = self.cv2(x)
        return self.cv3(torch.cat([main, skip], dim=1))


class CSPDarknet(nn.Module):
    """
    stem:   2 × CBS(3→32, 3×3)
    stage1: CBS(32→64, 3×3, s2) + CSPBlock(64→64,  n=3)
    stage2: CBS(64→128,3×3, s2) + CSPBlock(128→128,n=9)
    stage3: CBS(128→256,3×3,s2) + CSPBlock(256→256,n=9)
    stage4: CBS(256→512,3×3,s2) + CSPBlock(512→512,n=3)
    """
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(CBS(3, 32, 3, 1, 1), CBS(32, 32, 3, 1, 1))
        self.stage1 = nn.Sequential(CBS(32,  64,  3, 2, 1), CSPBlock(64,   64,  n=3))
        self.stage2 = nn.Sequential(CBS(64,  128, 3, 2, 1), CSPBlock(128,  128, n=9))
        self.stage3 = nn.Sequential(CBS(128, 256, 3, 2, 1), CSPBlock(256,  256, n=9))
        self.stage4 = nn.Sequential(CBS(256, 512, 3, 2, 1), CSPBlock(512,  512, n=3))

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stage1(x)
        p3 = self.stage2(x)   # [B, 128,  H/8,  W/8]
        p4 = self.stage3(p3)  # [B, 256,  H/16, W/16]
        p5 = self.stage4(p4)  # [B, 512,  H/32, W/32]
        return p3, p4, p5


# ─── CrossModalAttention ─────────────────────────────────────────────────────
# Checkpoint keys: cmaX.proj_sar / proj_opt / q_proj / k_proj / v_proj /
#                  o_proj / norm_sar / norm_opt / merge

class CrossModalAttention(nn.Module):
    """
    Projects SAR and optical features to feat_ch,
    applies cross-attention (SAR queries optical),
    then merges.
    """
    def __init__(self, sar_in, opt_in, feat_ch=256):
        super().__init__()
        self.proj_sar = CBS(sar_in,  feat_ch)   # .proj_sar.0 / .proj_sar.1
        self.proj_opt = CBS(opt_in,  feat_ch)   # .proj_opt.0 / .proj_opt.1

        self.q_proj = nn.Linear(feat_ch, feat_ch, bias=False)
        self.k_proj = nn.Linear(feat_ch, feat_ch, bias=False)
        self.v_proj = nn.Linear(feat_ch, feat_ch, bias=False)
        self.o_proj = nn.Linear(feat_ch, feat_ch, bias=False)

        self.norm_sar = nn.LayerNorm(feat_ch)
        self.norm_opt = nn.LayerNorm(feat_ch)

        self.merge = CBS(feat_ch * 2, feat_ch)  # .merge.0 / .merge.1
        self.feat_ch = feat_ch

    def forward(self, fs, fo):
        # Align spatial dims
        if fs.shape[2:] != fo.shape[2:]:
            fs = F.interpolate(fs, size=fo.shape[2:], mode='bilinear', align_corners=False)

        fs = self.proj_sar(fs)  # [B, C, H, W]
        fo = self.proj_opt(fo)

        B, C, H, W = fs.shape
        # Flatten spatial for attention
        qs = fs.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        ko = fo.flatten(2).permute(0, 2, 1)
        vo = ko

        qs = self.norm_sar(qs)
        ko = self.norm_opt(ko)

        Q = self.q_proj(qs)
        K = self.k_proj(ko)
        V = self.v_proj(vo)

        scale = math.sqrt(C)
        attn  = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1)
        out   = self.o_proj(torch.bmm(attn, V))  # [B, HW, C]
        out   = out.permute(0, 2, 1).view(B, C, H, W)

        return self.merge(torch.cat([fs, out], dim=1))


# ─── FPN Neck ────────────────────────────────────────────────────────────────
# Checkpoint keys: fusion.fpn.lat5 / lat4 / lat3  (Conv2d, no BN — 1×1 lateral)
#                  fusion.fpn.out5 / out4 / out3   (CBS: out.0 weight, out.1 BN)

class FPNNeck(nn.Module):
    def __init__(self, feat_ch=256):
        super().__init__()
        self.lat5 = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        self.lat4 = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        self.lat3 = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        self.out5 = CBS(feat_ch, feat_ch, 3, 1, 1)
        self.out4 = CBS(feat_ch, feat_ch, 3, 1, 1)
        self.out3 = CBS(feat_ch, feat_ch, 3, 1, 1)

    def forward(self, p3, p4, p5):
        f5 = self.out5(self.lat5(p5))
        f4 = self.out4(self.lat4(p4) + F.interpolate(f5, size=p4.shape[2:], mode='nearest'))
        f3 = self.out3(self.lat3(p3) + F.interpolate(f4, size=p3.shape[2:], mode='nearest'))
        return f3, f4, f5


# ─── Fusion Module ────────────────────────────────────────────────────────────
# Module name in checkpoint: "fusion"
# Contains: cma3, cma4, cma5, fpn

class FusionNeck(nn.Module):
    def __init__(self, feat_ch=256):
        super().__init__()
        # SAR channels from ResNet50:  512 / 1024 / 2048
        # Opt channels from CSPDarknet: 128 /  256 /  512
        self.cma3 = CrossModalAttention(512,  128, feat_ch)
        self.cma4 = CrossModalAttention(1024, 256, feat_ch)
        self.cma5 = CrossModalAttention(2048, 512, feat_ch)
        self.fpn  = FPNNeck(feat_ch)

    def forward(self, fs3, fs4, fs5, fo3, fo4, fo5):
        p3 = self.cma3(fs3, fo3)
        p4 = self.cma4(fs4, fo4)
        p5 = self.cma5(fs5, fo5)
        return self.fpn(p3, p4, p5)


# ─── Detection Head ──────────────────────────────────────────────────────────
# Checkpoint keys: det_head.headN.anchors / conv1 / conv2 / pred
# conv1 and conv2 are CBS (stored as .0 weight, .1 BN)

class YOLOHead(nn.Module):
    def __init__(self, in_ch, anchors, num_classes=1):
        super().__init__()
        na  = len(anchors)
        out = na * (5 + num_classes)
        self.register_buffer('anchors', torch.tensor(anchors, dtype=torch.float32))
        self.conv1 = CBS(in_ch,  in_ch * 2, 3, 1, 1)
        self.conv2 = CBS(in_ch * 2, in_ch,  1)
        self.pred  = nn.Conv2d(in_ch, out, 1)

    def forward(self, x):
        return self.pred(self.conv2(self.conv1(x)))


class MultiScaleHead(nn.Module):
    def __init__(self, feat_ch=256, num_classes=1):
        super().__init__()
        anchors3 = [(6,4),   (10,6),  (14,8)]
        anchors4 = [(20,12), (28,18), (36,24)]
        anchors5 = [(50,30), (70,45), (90,60)]
        self.head3 = YOLOHead(feat_ch, anchors3, num_classes)
        self.head4 = YOLOHead(feat_ch, anchors4, num_classes)
        self.head5 = YOLOHead(feat_ch, anchors5, num_classes)

    def forward(self, f3, f4, f5):
        return self.head3(f3), self.head4(f4), self.head5(f5)


# ─── Top-level model ─────────────────────────────────────────────────────────

class MultimodalDetector(nn.Module):
    """
    Top-level model whose state_dict keys exactly match best_model.pth:
      sar_backbone.*
      opt_backbone.*
      fusion.*        (fusion.cma3/4/5.* + fusion.fpn.*)
      det_head.*      (det_head.head3/4/5.*)
    """
    FEAT_CH     = 256
    NUM_CLASSES = 1

    def __init__(self, pretrained=False):
        super().__init__()
        self.sar_backbone = SARBackbone()
        self.opt_backbone = CSPDarknet()
        self.fusion       = FusionNeck(self.FEAT_CH)
        self.det_head     = MultiScaleHead(self.FEAT_CH, self.NUM_CLASSES)

    def forward(self, sar, opt):
        fs3, fs4, fs5 = self.sar_backbone(sar)
        fo3, fo4, fo5 = self.opt_backbone(opt)
        f3, f4, f5    = self.fusion(fs3, fs4, fs5, fo3, fo4, fo5)
        return self.det_head(f3, f4, f5)


# ─── Post-processing ─────────────────────────────────────────────────────────

def decode_predictions(pred: torch.Tensor, anchors: list, stride: int,
                       img_size: int, num_classes: int) -> list:
    """Decode raw YOLO prediction tensor into detection dicts."""
    B, C, H, W = pred.shape
    na = len(anchors)
    pred = pred.view(B, na, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()

    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    detections = []
    for b in range(B):
        for ai, (aw, ah) in enumerate(anchors):
            px  = (torch.sigmoid(pred[b, ai, :, :, 0]) + grid_x.float()) * stride
            py  = (torch.sigmoid(pred[b, ai, :, :, 1]) + grid_y.float()) * stride
            pw  = torch.exp(pred[b, ai, :, :, 2].clamp(-4, 4)) * aw
            ph  = torch.exp(pred[b, ai, :, :, 3].clamp(-4, 4)) * ah
            obj = torch.sigmoid(pred[b, ai, :, :, 4])
            cls = torch.sigmoid(pred[b, ai, :, :, 5])
            conf = obj * cls

            for cy, cx in (conf > 0.01).nonzero(as_tuple=False):
                cy, cx = cy.item(), cx.item()
                c  = float(conf[cy, cx])
                x1 = max(0.0, float(px[cy, cx]) - float(pw[cy, cx]) / 2)
                y1 = max(0.0, float(py[cy, cx]) - float(ph[cy, cx]) / 2)
                x2 = min(float(img_size), float(px[cy, cx]) + float(pw[cy, cx]) / 2)
                y2 = min(float(img_size), float(py[cy, cx]) + float(ph[cy, cx]) / 2)
                detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": c})
    return detections


def _iou(a: dict, b: dict) -> float:
    x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    ab = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections: list, iou_threshold=0.45,
                         conf_threshold=0.50) -> list:
    dets = [d for d in detections if d["confidence"] >= conf_threshold]
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best, d) < iou_threshold]
    return keep