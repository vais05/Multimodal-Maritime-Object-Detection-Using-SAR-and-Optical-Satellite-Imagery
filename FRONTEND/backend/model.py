"""
MarineVision model.py — EXACT architecture match for best_model.pth

Generated from Phase 4A / Phase 5 / Phase 6 / Phase 7 notebooks.
Every class name, attribute name, and layer structure is identical to
the training code so that load_state_dict(strict=True) succeeds.

Key differences from the previous backend model.py:
  1. ConvBnSilu wraps via self.conv = nn.Sequential(...)
     → checkpoint keys are  <name>.conv.0.*  <name>.conv.1.*
  2. SARBackbone uses torchvision ResNet-50 (stem = conv1+bn1+relu+maxpool)
     → checkpoint keys match torchvision's layout
  3. DetHead uses two independent 3×3 convs (in_ch→in_ch, in_ch→in_ch)
     → NOT 256→512 / 512→256
  4. decode_predictions returns normalised coords [0, 1]
     → main.py must scale by image dimensions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Constants (must match Phase 7 training)
# ──────────────────────────────────────────────────────────────────────────────
NUM_ANCHORS = 3
NUM_CLASSES = 1
FEAT_CH     = 256

ANCHORS = {
    'P3': [(6, 4),   (10, 6),  (14, 8)],
    'P4': [(20, 12), (28, 18), (36, 24)],
    'P5': [(50, 30), (70, 45), (90, 60)],
}
STRIDES = {'P3': 8, 'P4': 16, 'P5': 32}

SAR_CH = (512, 1024, 2048)
OPT_CH = (256, 512,  1024)


# ──────────────────────────────────────────────────────────────────────────────
# Building block — ConvBnSilu
# CRITICAL: self.conv wraps the Sequential so checkpoint keys are
#   <name>.conv.0.weight  (Conv2d)
#   <name>.conv.1.weight  (BatchNorm2d)
#   <name>.conv.1.running_mean  …
# ──────────────────────────────────────────────────────────────────────────────
class ConvBnSilu(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# CSP Bottleneck + CSP Layer (OpticalBackbone / CSPDarknet)
# ──────────────────────────────────────────────────────────────────────────────
class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=True):
        super().__init__()
        h = out_ch // 2
        self.cv1 = ConvBnSilu(in_ch, h, 1)
        self.cv2 = ConvBnSilu(h, out_ch, 3, 1, 1)
        self.sc  = shortcut and in_ch == out_ch

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.sc else self.cv2(self.cv1(x))


class CSPLayer(nn.Module):
    def __init__(self, in_ch, out_ch, n=1):
        super().__init__()
        h = out_ch // 2
        self.cv1 = ConvBnSilu(in_ch, h)
        self.cv2 = ConvBnSilu(in_ch, h)
        self.bns = nn.Sequential(*[Bottleneck(h, h) for _ in range(n)])
        self.cv3 = ConvBnSilu(2 * h, out_ch)

    def forward(self, x):
        return self.cv3(torch.cat([self.bns(self.cv1(x)), self.cv2(x)], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# SAR Backbone — torchvision ResNet-50, 1-channel input
# Uses torchvision's exact key layout: stem / layer1 / layer2 / layer3 / layer4
# ──────────────────────────────────────────────────────────────────────────────
class SARBackbone(nn.Module):
    def __init__(self, pretrained=False, freeze_early=False):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # Adapt first conv to 1 channel
        old = base.conv1
        new = nn.Conv2d(1, old.out_channels, old.kernel_size,
                        old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        base.conv1 = new

        # Store exactly as Phase 7 did
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        if freeze_early:
            for p in list(self.stem.parameters()) + list(self.layer1.parameters()):
                p.requires_grad = False

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        s2 = self.layer2(x)   # [B, 512,  H/8,  W/8]
        s3 = self.layer3(s2)  # [B, 1024, H/16, W/16]
        s4 = self.layer4(s3)  # [B, 2048, H/32, W/32]
        return s2, s3, s4

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


# ──────────────────────────────────────────────────────────────────────────────
# Optical Backbone — CSPDarknet
# ──────────────────────────────────────────────────────────────────────────────
class OpticalBackbone(nn.Module):
    def __init__(self, freeze_early=False):
        super().__init__()
        self.stem   = nn.Sequential(ConvBnSilu(3,   32,  3, 1, 1),
                                    ConvBnSilu(32,  64,  3, 2, 1))
        self.stage1 = nn.Sequential(ConvBnSilu(64,  128, 3, 2, 1), CSPLayer(128,  128,  3))
        self.stage2 = nn.Sequential(ConvBnSilu(128, 256, 3, 2, 1), CSPLayer(256,  256,  9))
        self.stage3 = nn.Sequential(ConvBnSilu(256, 512, 3, 2, 1), CSPLayer(512,  512,  9))
        self.stage4 = nn.Sequential(ConvBnSilu(512, 1024,3, 2, 1), CSPLayer(1024, 1024, 3))

        if freeze_early:
            for p in list(self.stem.parameters()) + list(self.stage1.parameters()):
                p.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.stage1(x)
        P3 = self.stage2(x)   # [B, 256,  H/8,  W/8]
        P4 = self.stage3(P3)  # [B, 512,  H/16, W/16]
        P5 = self.stage4(P4)  # [B, 1024, H/32, W/32]
        return P3, P4, P5

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Modal Attention
# proj_sar and proj_opt are nn.Sequential → keys: proj_sar.0.* / proj_sar.1.*
# merge is nn.Sequential → keys: merge.0.* / merge.1.*
# ──────────────────────────────────────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    def __init__(self, sar_ch, opt_ch, out_ch=256, attn_dim=128, pool_size=4):
        super().__init__()
        self.attn_dim  = attn_dim
        self.pool_size = pool_size
        self.scale     = attn_dim ** -0.5

        self.proj_sar = nn.Sequential(
            nn.Conv2d(sar_ch, attn_dim, 1, bias=False),
            nn.BatchNorm2d(attn_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_opt = nn.Sequential(
            nn.Conv2d(opt_ch, attn_dim, 1, bias=False),
            nn.BatchNorm2d(attn_dim),
            nn.ReLU(inplace=True),
        )

        self.q_proj = nn.Linear(attn_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(attn_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(attn_dim, attn_dim, bias=False)
        self.o_proj = nn.Linear(attn_dim, attn_dim, bias=False)

        self.norm_sar = nn.LayerNorm(attn_dim)
        self.norm_opt = nn.LayerNorm(attn_dim)

        self.merge = nn.Sequential(
            nn.Conv2d(2 * attn_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, sar_feat, opt_feat):
        B, _, H, W = sar_feat.shape
        sp = self.proj_sar(sar_feat)
        op = self.proj_opt(opt_feat)

        ps = max(1, self.pool_size)
        op_pool = F.avg_pool2d(op, ps, stride=ps, padding=0)

        Q  = self.q_proj(self.norm_sar(sp.flatten(2).transpose(1, 2)))
        KV = self.norm_opt(op_pool.flatten(2).transpose(1, 2))
        K  = self.k_proj(KV)
        V  = self.v_proj(KV)

        att = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        out = self.o_proj(torch.bmm(att, V))
        out = out.transpose(1, 2).reshape(B, self.attn_dim, H, W)

        return self.merge(torch.cat([sp, out], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# FPN Neck
# out3/out4/out5 are nn.Sequential → keys: out3.0.* / out3.1.*
# ──────────────────────────────────────────────────────────────────────────────
class FPNNeck(nn.Module):
    def __init__(self, feat_ch=256):
        super().__init__()
        self.lat5 = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        self.lat4 = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        self.lat3 = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)

        def _out(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            )

        self.out5 = _out(feat_ch)
        self.out4 = _out(feat_ch)
        self.out3 = _out(feat_ch)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, P3, P4, P5):
        l5 = self.lat5(P5)
        l4 = self.lat4(P4)
        l3 = self.lat3(P3)
        o5 = self.out5(l5)
        o4 = self.out4(l4 + F.interpolate(o5, size=l4.shape[2:], mode='nearest'))
        o3 = self.out3(l3 + F.interpolate(o4, size=l3.shape[2:], mode='nearest'))
        return o3, o4, o5


# ──────────────────────────────────────────────────────────────────────────────
# Fusion Neck
# ──────────────────────────────────────────────────────────────────────────────
class FusionNeck(nn.Module):
    def __init__(self, sar_ch=(512, 1024, 2048),
                 opt_ch=(256, 512, 1024),
                 feat_ch=256, attn_dim=128):
        super().__init__()
        self.cma3 = CrossModalAttention(sar_ch[0], opt_ch[0], feat_ch, attn_dim, pool_size=4)
        self.cma4 = CrossModalAttention(sar_ch[1], opt_ch[1], feat_ch, attn_dim, pool_size=2)
        self.cma5 = CrossModalAttention(sar_ch[2], opt_ch[2], feat_ch, attn_dim, pool_size=1)
        self.fpn  = FPNNeck(feat_ch)

    def forward(self, Fs3, Fs4, Fs5, Fo3, Fo4, Fo5):
        return self.fpn(
            self.cma3(Fs3, Fo3),
            self.cma4(Fs4, Fo4),
            self.cma5(Fs5, Fo5),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Detection Head — DetHead (single scale)
# conv1: Sequential(Conv2d in_ch→in_ch 3×3, BN, LeakyReLU)
# conv2: Sequential(Conv2d in_ch→in_ch 3×3, BN, LeakyReLU)
# pred:  Conv2d in_ch → num_anchors*(5+num_classes)
#
# Keys: conv1.0.weight / conv1.1.* / conv2.0.weight / conv2.1.*
# ──────────────────────────────────────────────────────────────────────────────
class DetHead(nn.Module):
    def __init__(self, in_ch, num_anchors, num_classes, anchors, stride):
        super().__init__()
        pred_ch = num_anchors * (4 + 1 + num_classes)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.stride      = stride
        self.pred_ch     = pred_ch
        self.register_buffer('anchors',
                             torch.tensor(anchors, dtype=torch.float32))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pred = nn.Conv2d(in_ch, pred_ch, 1, bias=True)

        # Bias init: objectness starts at ~0.01
        bias_init = -math.log((1 - 0.01) / 0.01)
        with torch.no_grad():
            for i in range(num_anchors):
                self.pred.bias[i * (4 + 1 + num_classes) + 4] = bias_init

    def forward(self, x):
        return self.pred(self.conv2(self.conv1(x)))


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Scale Head
# ──────────────────────────────────────────────────────────────────────────────
class MultiScaleHead(nn.Module):
    def __init__(self, feat_ch=256, num_anchors=3, num_classes=1,
                 anchors=None, strides=None):
        super().__init__()
        if anchors is None:
            anchors = ANCHORS
        if strides is None:
            strides = STRIDES
        self.head3 = DetHead(feat_ch, num_anchors, num_classes,
                             anchors['P3'], strides['P3'])
        self.head4 = DetHead(feat_ch, num_anchors, num_classes,
                             anchors['P4'], strides['P4'])
        self.head5 = DetHead(feat_ch, num_anchors, num_classes,
                             anchors['P5'], strides['P5'])

    def forward(self, P3, P4, P5):
        return self.head3(P3), self.head4(P4), self.head5(P5)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level model
# ──────────────────────────────────────────────────────────────────────────────
class MultimodalDetector(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.sar_backbone = SARBackbone(pretrained=pretrained, freeze_early=False)
        self.opt_backbone = OpticalBackbone(freeze_early=False)
        self.fusion       = FusionNeck(sar_ch=SAR_CH, opt_ch=OPT_CH, feat_ch=FEAT_CH)
        self.det_head     = MultiScaleHead(FEAT_CH, NUM_ANCHORS, NUM_CLASSES)

    def forward(self, sar, opt):
        Fs3, Fs4, Fs5 = self.sar_backbone(sar)
        Fo3, Fo4, Fo5 = self.opt_backbone(opt)
        P3,  P4,  P5  = self.fusion(Fs3, Fs4, Fs5, Fo3, Fo4, Fo5)
        return self.det_head(P3, P4, P5)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing — exact copy of Phase 8 evaluation decode logic
# Returns normalised coordinates [0, 1]  (main.py scales to pixels)
# ──────────────────────────────────────────────────────────────────────────────
def decode_predictions(preds, anchors_list, strides_list, device,
                       conf_thresh=0.05, nms_thresh=0.45,
                       img_h=320, img_w=320, max_preds=1000):
    """
    Decode raw model outputs into bounding-box detections.

    preds        : tuple of (pred_P3, pred_P4, pred_P5) raw tensors
    anchors_list : list of anchor lists, one per scale
    strides_list : list of strides [8, 16, 32]
    Returns      : list (one per image) of dicts with normalised coords
                   {'x1','y1','x2','y2','confidence'}  all in [0, 1]
    """
    B = preds[0].shape[0]
    results = [[] for _ in range(B)]

    for pred, ancs, stride in zip(preds, anchors_list, strides_list):
        _, _, H, W = pred.shape
        A = NUM_ANCHORS
        C = NUM_CLASSES
        pr = pred.view(B, A, 4 + 1 + C, H, W)
        anc_t = torch.tensor(ancs, device=device, dtype=torch.float32)

        gy_grid, gx_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )

        for b in range(B):
            for ai in range(A):
                tx   = torch.sigmoid(pr[b, ai, 0])
                ty   = torch.sigmoid(pr[b, ai, 1])
                tw   = torch.exp(pr[b, ai, 2].clamp(-4, 4))
                th   = torch.exp(pr[b, ai, 3].clamp(-4, 4))
                obj  = torch.sigmoid(pr[b, ai, 4])
                cls  = torch.sigmoid(pr[b, ai, 5])
                conf = obj * cls

                mask = conf > conf_thresh
                if not mask.any():
                    continue

                gx_v = gx_grid[mask]
                gy_v = gy_grid[mask]
                bx   = (tx[mask] + gx_v) * stride / img_w   # normalised [0,1]
                by   = (ty[mask] + gy_v) * stride / img_h
                bw   = tw[mask] * anc_t[ai, 0] / img_w
                bh   = th[mask] * anc_t[ai, 1] / img_h
                c    = conf[mask]

                x1 = (bx - bw / 2).clamp(0, 1)
                y1 = (by - bh / 2).clamp(0, 1)
                x2 = (bx + bw / 2).clamp(0, 1)
                y2 = (by + bh / 2).clamp(0, 1)

                for i in range(len(c)):
                    results[b].append({
                        'x1': float(x1[i]),
                        'y1': float(y1[i]),
                        'x2': float(x2[i]),
                        'y2': float(y2[i]),
                        'confidence': float(c[i]),
                    })

    # NMS per image
    final = []
    for b in range(B):
        dets = results[b]
        if not dets:
            final.append([])
            continue
        dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)
        dets = dets[:max_preds]
        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [d for d in dets if _iou(best, d) < nms_thresh]
        final.append(keep)

    return final


def _iou(a, b):
    ix1 = max(a['x1'], b['x1']); iy1 = max(a['y1'], b['y1'])
    ix2 = min(a['x2'], b['x2']); iy2 = min(a['y2'], b['y2'])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    ab = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections, iou_threshold=0.45, conf_threshold=0.50):
    """Legacy helper kept for compatibility."""
    dets = [d for d in detections if d['confidence'] >= conf_threshold]
    dets.sort(key=lambda d: d['confidence'], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best, d) < iou_threshold]
    return keep