"""
phase7_model.py
All architecture classes extracted from PHASE7_Training_FIXED.ipynb.
Import with:  from phase7_model import MultimodalDetector
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# ── Constants ────────────────────────────────────────────────────────────────
FEAT_CH     = 256
NUM_ANCHORS = 3
NUM_CLASSES = 1
SAR_CH      = (512, 1024, 2048)
OPT_CH      = (256, 512,  1024)

ANCHORS = {
    'P3': [(6,4),   (10,6),  (14,8)],
    'P4': [(20,12), (28,18), (36,24)],
    'P5': [(50,30), (70,45), (90,60)],
}
STRIDES = {'P3': 8, 'P4': 16, 'P5': 32}


# ── Phase 4A: CSPDarknet building blocks ─────────────────────────────────────
class ConvBnSilu(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


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
        self.cv3 = ConvBnSilu(2*h, out_ch)

    def forward(self, x):
        return self.cv3(torch.cat([self.bns(self.cv1(x)), self.cv2(x)], 1))


# ── Phase 4A: SAR Backbone (ResNet50, 1-channel input) ───────────────────────
class SARBackbone(nn.Module):
    def __init__(self, pretrained=True, freeze_early=True):
        super().__init__()
        base = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        old = base.conv1
        new = nn.Conv2d(1, old.out_channels,
                        old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        base.conv1  = new
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


# ── Phase 4A: Optical Backbone (CSPDarknet, 3-channel input) ─────────────────
class OpticalBackbone(nn.Module):
    def __init__(self, freeze_early=True):
        super().__init__()
        self.stem   = nn.Sequential(ConvBnSilu(3, 32, 3, 1, 1),
                                    ConvBnSilu(32, 64, 3, 2, 1))
        self.stage1 = nn.Sequential(ConvBnSilu(64, 128, 3, 2, 1),  CSPLayer(128, 128, 3))
        self.stage2 = nn.Sequential(ConvBnSilu(128, 256, 3, 2, 1), CSPLayer(256, 256, 9))
        self.stage3 = nn.Sequential(ConvBnSilu(256, 512, 3, 2, 1), CSPLayer(512, 512, 9))
        self.stage4 = nn.Sequential(ConvBnSilu(512, 1024, 3, 2, 1),CSPLayer(1024,1024, 3))
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


# ── Phase 5: Cross-Modal Attention ───────────────────────────────────────────
class CrossModalAttention(nn.Module):
    def __init__(self, sar_ch, opt_ch, out_ch=256, attn_dim=128, pool_size=4):
        super().__init__()
        self.attn_dim  = attn_dim
        self.pool_size = pool_size
        self.scale     = attn_dim ** -0.5
        self.proj_sar  = nn.Sequential(
            nn.Conv2d(sar_ch, attn_dim, 1, bias=False),
            nn.BatchNorm2d(attn_dim), nn.ReLU(inplace=True))
        self.proj_opt  = nn.Sequential(
            nn.Conv2d(opt_ch, attn_dim, 1, bias=False),
            nn.BatchNorm2d(attn_dim), nn.ReLU(inplace=True))
        self.q_proj   = nn.Linear(attn_dim, attn_dim, bias=False)
        self.k_proj   = nn.Linear(attn_dim, attn_dim, bias=False)
        self.v_proj   = nn.Linear(attn_dim, attn_dim, bias=False)
        self.o_proj   = nn.Linear(attn_dim, attn_dim, bias=False)
        self.norm_sar = nn.LayerNorm(attn_dim)
        self.norm_opt = nn.LayerNorm(attn_dim)
        self.merge    = nn.Sequential(
            nn.Conv2d(2*attn_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, sar_feat, opt_feat):
        B, _, H, W = sar_feat.shape
        sp      = self.proj_sar(sar_feat)
        op      = self.proj_opt(opt_feat)
        op_pool = F.avg_pool2d(op, max(1, self.pool_size),
                               stride=max(1, self.pool_size), padding=0)
        Q  = self.q_proj(self.norm_sar(sp.flatten(2).transpose(1, 2)))
        KV = self.norm_opt(op_pool.flatten(2).transpose(1, 2))
        K  = self.k_proj(KV)
        V  = self.v_proj(KV)
        att = F.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        out = self.o_proj(torch.bmm(att, V)).transpose(1, 2).reshape(
            B, self.attn_dim, H, W)
        return self.merge(torch.cat([sp, out], dim=1))


# ── Phase 5: FPN Neck ────────────────────────────────────────────────────────
class FPNNeck(nn.Module):
    def __init__(self, feat_ch=256):
        super().__init__()
        self.lat5 = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        self.lat4 = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        self.lat3 = nn.Conv2d(feat_ch, feat_ch, 1, bias=False)
        def _out(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c), nn.ReLU(inplace=True))
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


# ── Phase 5: Fusion Neck (CMA × 3 + FPN) ────────────────────────────────────
class FusionNeck(nn.Module):
    def __init__(self, sar_ch=(512,1024,2048), opt_ch=(256,512,1024),
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
            self.cma5(Fs5, Fo5))


# ── Phase 6: Detection Heads ─────────────────────────────────────────────────
class DetHead(nn.Module):
    def __init__(self, in_ch, num_anchors, num_classes, anchors, stride):
        super().__init__()
        pred = num_anchors * (4 + 1 + num_classes)
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.stride      = stride
        self.pred_ch     = pred
        self.register_buffer('anchors',
                             torch.tensor(anchors, dtype=torch.float32))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch), nn.LeakyReLU(0.1, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch), nn.LeakyReLU(0.1, inplace=True))
        self.pred  = nn.Conv2d(in_ch, pred, 1, bias=True)
        bias_init  = -math.log((1 - 0.01) / 0.01)
        with torch.no_grad():
            for i in range(num_anchors):
                self.pred.bias[i * (4 + 1 + num_classes) + 4] = bias_init

    def forward(self, x):
        return self.pred(self.conv2(self.conv1(x)))


class MultiScaleHead(nn.Module):
    def __init__(self, feat_ch=256, num_anchors=3, num_classes=1,
                 anchors=ANCHORS, strides=STRIDES):
        super().__init__()
        self.head3 = DetHead(feat_ch, num_anchors, num_classes,
                             anchors['P3'], strides['P3'])
        self.head4 = DetHead(feat_ch, num_anchors, num_classes,
                             anchors['P4'], strides['P4'])
        self.head5 = DetHead(feat_ch, num_anchors, num_classes,
                             anchors['P5'], strides['P5'])

    def forward(self, P3, P4, P5):
        return self.head3(P3), self.head4(P4), self.head5(P5)


# ── Complete Model ────────────────────────────────────────────────────────────
class MultimodalDetector(nn.Module):
    """
    Complete multimodal ship detection model.
    Combines Phase 4A backbone + Phase 5 fusion + Phase 6 head.

    Inputs:
        sar : [B, 1, H, W]   — single-channel SAR image
        opt : [B, 3, H, W]   — three-channel optical image
    Outputs:
        (pred_P3, pred_P4, pred_P5)
        each shape [B, 18, H/stride, W/stride]   (18 = 3 anchors × 6 values)
    """
    def __init__(self, pretrained=False):
        super().__init__()
        self.sar_backbone = SARBackbone(pretrained=pretrained, freeze_early=True)
        self.opt_backbone = OpticalBackbone(freeze_early=True)
        self.fusion       = FusionNeck(sar_ch=SAR_CH, opt_ch=OPT_CH, feat_ch=FEAT_CH)
        self.det_head     = MultiScaleHead(FEAT_CH, NUM_ANCHORS, NUM_CLASSES)

    def forward(self, sar, opt):
        Fs3, Fs4, Fs5 = self.sar_backbone(sar)
        Fo3, Fo4, Fo5 = self.opt_backbone(opt)
        oP3, oP4, oP5 = self.fusion(Fs3, Fs4, Fs5, Fo3, Fo4, Fo5)
        return self.det_head(oP3, oP4, oP5)

    def unfreeze_backbones(self):
        self.sar_backbone.unfreeze()
        self.opt_backbone.unfreeze()

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, train
