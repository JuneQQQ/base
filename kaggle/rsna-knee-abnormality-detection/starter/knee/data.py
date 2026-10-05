"""Training data: per-study cached stacks -> bags of 2.5D windows, plus GPU augmentation."""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import PrepConfig
from .model import window_index


def load_prep(cache_dir: str) -> PrepConfig:
    with open(os.path.join(cache_dir, "prep.json")) as f:
        d = json.load(f)
    d["slots"] = tuple(tuple(s) for s in d["slots"])
    d["span"] = tuple(d["span"])
    return PrepConfig(**d)


def load_masks(cache_dir: str) -> dict:
    """uid -> (S,) uint8 slice validity, from the meta_*.csv files written by 01_build_cache."""
    out = {}
    for p in glob.glob(os.path.join(cache_dir, "meta_*.csv")):
        m = pd.read_csv(p, dtype={"StudyInstanceUID": str, "mask": str})
        for uid, s in zip(m["StudyInstanceUID"], m["mask"]):
            out[uid] = np.frombuffer(str(s).encode(), np.uint8) - ord("0")
    return out


class StudyBags(Dataset):
    """k random valid windows per study when training, every window when evaluating."""

    def __init__(self, uids, cache_dir, prep: PrepConfig, labels=None, weights=None,
                 k: int | None = 16, masks: dict | None = None):
        self.uids = list(uids)
        self.cache_dir = cache_dir
        self.centres, self.slot_ids = window_index(prep)
        self.labels = labels
        self.weights = weights
        self.k = k
        self.masks = masks or {}
        self.S = prep.n_slices

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        vol = np.load(os.path.join(self.cache_dir, f"{uid}.npy"), mmap_mode="r")
        mask = self.masks.get(uid)
        if mask is None or len(mask) != self.S:
            mask = (np.asarray(vol).reshape(self.S, -1).max(1) > 0).astype(np.uint8)
        ok = mask[self.centres - 1] & mask[self.centres] & mask[self.centres + 1]
        if self.k is None:
            sel = np.arange(len(self.centres))
        else:
            pool = np.flatnonzero(ok)
            if len(pool) == 0:
                pool = np.arange(len(self.centres))
            sel = np.random.choice(pool, self.k, replace=len(pool) < self.k)
        c = self.centres[sel]
        idx = np.stack([c - 1, c, c + 1], 1)
        wins = np.asarray(vol[idx.ravel()]).reshape(len(sel), 3, *vol.shape[1:])
        out = {"x": torch.from_numpy(wins), "slot": torch.from_numpy(self.slot_ids[sel]),
               "ok": torch.from_numpy(ok[sel].astype(bool))}
        if self.labels is not None:
            out["y"] = torch.from_numpy(self.labels[i].astype(np.float32))
            out["w"] = torch.from_numpy(self.weights[i].astype(np.float32))
        return out


def gpu_augment(x: torch.Tensor, img: int, train: bool, max_rot=10.0, scale=(0.9, 1.1),
                shift=0.05, intensity=0.15) -> torch.Tensor:
    """x (B,K,3,H,W) uint8 on GPU -> float (B,K,3,img,img) in [0,1].

    One random affine per study (all windows of a study move together). No flips: after
    laterality normalisation, left/right position IS the medial/lateral signal.
    """
    B, K, C, H, W = x.shape
    x = x.float().div_(255).flatten(0, 1)
    if not train:
        return F.interpolate(x, size=(img, img), mode="bilinear", align_corners=False).view(B, K, C, img, img)
    dev = x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(max_rot)
    sc = torch.empty(B, device=dev).uniform_(*scale)
    tx, ty = [(torch.rand(B, device=dev) * 2 - 1) * shift * 2 for _ in range(2)]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    theta = theta.repeat_interleave(K, 0)
    grid = F.affine_grid(theta, (B * K, C, img, img), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # brightness / contrast / gamma, per study
    g = torch.exp((torch.rand(B, device=dev) * 2 - 1) * intensity).repeat_interleave(K).view(-1, 1, 1, 1)
    ct = (1 + (torch.rand(B, device=dev) * 2 - 1) * intensity).repeat_interleave(K).view(-1, 1, 1, 1)
    br = ((torch.rand(B, device=dev) * 2 - 1) * intensity / 2).repeat_interleave(K).view(-1, 1, 1, 1)
    x = (x.clamp(0, 1) ** g - 0.5) * ct + 0.5 + br
    return x.clamp_(0, 1).view(B, K, C, img, img)
