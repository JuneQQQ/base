"""2.5D study-level classifier.

A study stack (S, H, W) is cut into windows of 3 physically adjacent slices (never crossing
a slot boundary); each window is one 3-channel image for a 2D ImageNet backbone. Window
features get a learned slot (plane/sequence) embedding and are pooled by attention with a
separate softmax per finding (an ACL tear lives on 2-3 sagittal slices, OA spreads over
many coronal ones, so one shared pooling makes the findings compete), plus a masked-mean
residual head for stability.
"""
from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_PREP, N_LABELS, PrepConfig


def window_index(prep: PrepConfig = DEFAULT_PREP):
    """All (centre slice, slot id) pairs whose 3-slice window stays inside one slot."""
    centres, slots = [], []
    for s, (a, b) in enumerate(prep.slot_bounds()):
        for c in range(a + 1, b - 1):
            centres.append(c)
            slots.append(s)
    return np.asarray(centres), np.asarray(slots)


def gather_windows(vol: np.ndarray, mask: np.ndarray, centres: np.ndarray):
    """vol (S,H,W) uint8 -> (K,3,H,W) uint8 and (K,) bool validity (all 3 slices present)."""
    idx = np.stack([centres - 1, centres, centres + 1], 1)
    wins = vol[idx]                       # (K,3,H,W)
    ok = mask[idx].all(1).astype(bool)
    return wins, ok


class KneeNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, n_slots: int = 5,
                 n_labels: int = N_LABELS, drop: float = 0.2, drop_path: float = 0.1):
        super().__init__()
        kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
        try:
            self.backbone = timm.create_model(backbone, drop_path_rate=drop_path, **kw)
        except TypeError:  # some archs don't take drop_path_rate
            self.backbone = timm.create_model(backbone, **kw)
        cfg = self.backbone.pretrained_cfg
        # a checkpoint carries its own preprocessing contract; read it, don't hardcode ImageNet
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        fd = self.backbone.num_features
        self.slot_emb = nn.Embedding(n_slots, fd)
        nn.init.normal_(self.slot_emb.weight, std=0.02)
        self.norm = nn.LayerNorm(fd)
        self.att = nn.Sequential(nn.Linear(fd, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n_labels))
        self.cls_w = nn.Parameter(torch.empty(n_labels, fd))
        nn.init.trunc_normal_(self.cls_w, std=0.02)
        self.cls_b = nn.Parameter(torch.zeros(n_labels))
        self.mean_head = nn.Sequential(nn.Dropout(drop), nn.Linear(fd, n_labels))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,K,3,H,W) float in [0,1] -> (B,K,F)."""
        B, K = x.shape[:2]
        x = (x.flatten(0, 1) - self.mean) / self.std
        return self.backbone(x).view(B, K, -1)

    def head(self, feats, slot_ids, win_mask):
        h = self.norm(feats + self.slot_emb(slot_ids))
        m = win_mask.unsqueeze(-1)                                   # (B,K,1)
        a = self.att(h).float().masked_fill(~m, -1e4)                # (B,K,L)
        a = torch.softmax(a, dim=1).to(h.dtype)
        pooled = torch.einsum("bkl,bkf->blf", a, h)                  # (B,L,F)
        logits = (pooled * self.cls_w).sum(-1) + self.cls_b
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return logits + self.mean_head(mean)

    def forward(self, x, slot_ids, win_mask):
        return self.head(self.encode(x), slot_ids, win_mask)


def load_checkpoint(path: str, device="cpu") -> tuple[KneeNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = KneeNet(ck["backbone"], pretrained=False, n_slots=ck.get("n_slots", 5))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_study(model: KneeNet, vol: np.ndarray, mask: np.ndarray, img: int,
                  centres: np.ndarray, slot_ids: np.ndarray, device, dtype=torch.float16,
                  tta_shift: bool = False) -> np.ndarray:
    """Sigmoid probabilities (L,) for one study stack."""
    wins, ok = gather_windows(vol, mask, centres)
    if not ok.any():
        return np.full(N_LABELS, np.nan, np.float32)
    x = torch.from_numpy(wins).to(device, non_blocking=True).float().div_(255)
    if x.shape[-1] != img:
        x = F.interpolate(x, size=(img, img), mode="bilinear", align_corners=False)
    x = x.unsqueeze(0)
    sid = torch.from_numpy(slot_ids).to(device).unsqueeze(0)
    wm = torch.from_numpy(ok).to(device).unsqueeze(0)
    use_amp = device.type == "cuda" and dtype != torch.float32
    with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
        logits = model(x, sid, wm).float()
        if tta_shift:  # light zoom TTA (forum: up to 1.25x zoom is AUC-neutral for these models)
            z = F.interpolate(x.flatten(0, 1), scale_factor=1.1, mode="bilinear", align_corners=False)
            o = (z.shape[-1] - img) // 2
            z = z[..., o:o + img, o:o + img].reshape_as(x)
            logits = (logits + model(z, sid, wm).float()) / 2
    return torch.sigmoid(logits)[0].cpu().numpy()
