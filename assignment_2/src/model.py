"""Frozen ResNet backbone + attention-MIL temporal head."""
import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name="resnet50", pretrained=True):
    if pretrained:
        try:
            w = {"resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
                 "resnet18": models.ResNet18_Weights.IMAGENET1K_V1}[name]
        except Exception:
            return _legacy_backbone(name)
    else:
        w = None
    base = {"resnet50": models.resnet50, "resnet18": models.resnet18}[name](weights=w)
    mods = list(base.children())[:-2]  # drop avgpool + fc
    feat = nn.Sequential(*mods, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    dim = {"resnet50": 2048, "resnet18": 512}[name]
    for p in feat.parameters():
        p.requires_grad = False
    feat.eval()
    return feat, dim


def _legacy_backbone(name):
    base = {"resnet50": models.resnet50, "resnet18": models.resnet18}[name](pretrained=True)
    mods = list(base.children())[:-2]
    feat = nn.Sequential(*mods, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    dim = {"resnet50": 2048, "resnet18": 512}[name]
    for p in feat.parameters():
        p.requires_grad = False
    feat.eval()
    return feat, dim


class CrashModel(nn.Module):
    def __init__(self, feat_dim, hidden=512, dropout=0.3):
        super().__init__()
        self.backbone, _ = None, None  # set via factory below
        self.fc = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(),
                                nn.Dropout(dropout))
        self.attn = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, 1)

    def forward(self, feats):  # feats: (B, T, D) precomputed frame features
        h = self.fc(feats)                       # (B, T, H)
        w = torch.softmax(self.attn(h), dim=1)   # (B, T, 1)
        ctx = (h * w).sum(dim=1)                 # (B, H)
        return self.out(ctx).squeeze(1), w.squeeze(2)  # logits, attn weights


class VideoModel(nn.Module):
    """End-to-end wrapper: frames -> frozen backbone -> head."""

    def __init__(self, backbone_name="resnet50", pretrained=True,
                 hidden=512, dropout=0.3):
        super().__init__()
        self.backbone, dim = build_backbone(backbone_name, pretrained)
        self.head = CrashModel(dim, hidden, dropout)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T = x.shape[:2]
        with torch.no_grad():
            f = self.backbone(x.view(B * T, *x.shape[2:])).view(B, T, -1)
        return self.head(f)
