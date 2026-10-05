#!/usr/bin/env python3
"""Train one fold (or a full fit) of the 2.5D KneeNet on report-derived soft labels.

  python scripts/04_train.py --cache /data/cache_256 --labels labels.csv --folds folds.csv \
      --train-csv train.csv --fold 0 --backbone resnet34 --img 224 --epochs 15 --out runs/r34_f0

Validation, in order of trust (forum consensus):
  1. OOF macro AUC against the report labels (thousands of studies; measure your seed noise,
     ~0.002-0.003, and only keep changes above it),
  2. public LB to calibrate that CV -> LB mapping (it was strongly correlated for most teams),
  3. the 58 gold studies as a sanity check only (too small to rank two models).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from knee.config import KNOWN_BAD_STUDIES, LABELS  # noqa: E402
from knee.data import StudyBags, gpu_augment, load_masks, load_prep  # noqa: E402
from knee.model import KneeNet  # noqa: E402


def macro_auc(y, p, binarize=True):
    res = {}
    for j, lab in enumerate(LABELS):
        t = (y[:, j] >= 0.5).astype(int) if binarize else y[:, j].astype(int)
        if 0 < t.sum() < len(t):
            res[lab] = roc_auc_score(t, p[:, j])
    return float(np.mean(list(res.values()))) if res else float("nan"), res


def amp_dtype(device):
    if device.type != "cuda":
        return None
    # T4/P100 have no bf16 units; emulated bf16 is ~50x slower (discussion 744230)
    try:
        ok = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        ok = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if ok else torch.float16


@torch.no_grad()
def predict(model, loader, img, device, dtype):
    model.eval()
    out = []
    for b in loader:
        x = gpu_augment(b["x"].to(device, non_blocking=True), img, train=False)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            out.append(torch.sigmoid(model(x, b["slot"].to(device), b["ok"].to(device)).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--train-csv", required=True, help="for the 58 gold labels")
    ap.add_argument("--fold", type=int, default=0, help="-1 = train on all non-gold studies")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--k", type=int, default=16, help="windows sampled per study per step")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="backbone lr; head gets 10x")
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.0, help="e.g. 0.998; 0 disables")
    ap.add_argument("--gold-in-train", action="store_true", help="add the 58 gold studies (weight 3)")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training studies")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = amp_dtype(dev)
    prep = load_prep(a.cache)
    masks = load_masks(a.cache)
    have = {f[:-4] for f in os.listdir(a.cache) if f.endswith(".npy")}

    tr = pd.read_csv(a.train_csv)
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS]
    lab = pd.read_csv(a.labels, index_col=0)
    lab.index = lab.index.astype(str)
    wcols = [f"w_{c}" for c in LABELS]
    for c in wcols:
        if c not in lab:
            lab[c] = 1.0
    folds = pd.read_csv(a.folds, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")["fold"]

    ok_ids = [u for u in folds.index if u in have and u not in KNOWN_BAD_STUDIES and u in lab.index]
    tr_ids = [u for u in ok_ids if folds[u] != -1 and folds[u] != a.fold]
    va_ids = [u for u in ok_ids if folds[u] == a.fold] if a.fold >= 0 else []
    gold_ids = [u for u in gold.index if u in have]
    if a.limit:
        tr_ids = tr_ids[:a.limit]
    y_tr, w_tr = lab.loc[tr_ids, LABELS].values, lab.loc[tr_ids, wcols].values
    if a.gold_in_train:
        tr_ids += gold_ids
        y_tr = np.concatenate([y_tr, gold.loc[gold_ids].values])
        w_tr = np.concatenate([w_tr, np.full((len(gold_ids), len(LABELS)), 3.0)])
    print(f"device {dev} amp {dtype} | train {len(tr_ids)} | val {len(va_ids)} | gold {len(gold_ids)}", flush=True)

    ds_tr = StudyBags(tr_ids, a.cache, prep, y_tr, w_tr, k=a.k, masks=masks)
    dl_tr = DataLoader(ds_tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                       pin_memory=True, persistent_workers=a.workers > 0)
    mk = lambda ids: DataLoader(StudyBags(ids, a.cache, prep, k=None, masks=masks), batch_size=max(1, a.bs // 4),
                                num_workers=a.workers, pin_memory=True)
    dl_va, dl_gold = mk(va_ids), mk(gold_ids)

    model = KneeNet(a.backbone, pretrained=bool(a.pretrained), n_slots=len(prep.slots)).to(dev)
    model = model.to(memory_format=torch.channels_last)
    if a.grad_ckpt:
        model.backbone.set_grad_checkpointing(True)
    print("backbone:", a.backbone, "| norm mean/std", model.mean.flatten().tolist(), model.std.flatten().tolist())
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    hd = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr}, {"params": hd, "lr": a.lr * 10}], weight_decay=a.wd)
    steps = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 10], total_steps=max(steps, 1),
                                                pct_start=a.warmup, anneal_strategy="cos")
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    ema = None
    if a.ema > 0:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=a.ema)

    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in dl_tr:
            x = gpu_augment(b["x"].to(dev, non_blocking=True), a.img, train=True)
            y, w = b["y"].to(dev), b["w"].to(dev)
            with torch.autocast(dev.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                logits = model(x, b["slot"].to(dev), b["ok"].to(dev))
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item()
        net = ema.module if ema is not None else model
        rec = {"epoch": ep, "loss": tot / max(1, len(dl_tr)), "sec": round(time.time() - t0)}
        if va_ids:
            p_va = predict(net, dl_va, a.img, dev, dtype)
            rec["val_auc"], _ = macro_auc(lab.loc[va_ids, LABELS].values, p_va)
        if gold_ids and not a.gold_in_train:
            p_g = predict(net, dl_gold, a.img, dev, dtype)
            rec["gold_auc"], _ = macro_auc(gold.loc[gold_ids].values, p_g, binarize=False)
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        score = rec.get("val_auc", float(ep))  # full fit: keep the last epoch
        if not np.isfinite(score):  # no label has both classes in val: keep the latest epoch
            score = float(ep)
        if score >= best:
            best = score
            torch.save({"model": net.state_dict(), "backbone": a.backbone, "img": a.img,
                        "n_slots": len(prep.slots), "prep": json.load(open(os.path.join(a.cache, "prep.json"))),
                        "labels": LABELS, "epoch": ep, "val_auc": rec.get("val_auc")},
                       os.path.join(a.out, "best.pt"))
            if va_ids:
                pd.DataFrame(p_va, index=pd.Index(va_ids, name="StudyInstanceUID"), columns=LABELS).to_csv(
                    os.path.join(a.out, f"oof_fold{a.fold}.csv"))
            if gold_ids and not a.gold_in_train:
                pd.DataFrame(p_g, index=pd.Index(gold_ids, name="StudyInstanceUID"), columns=LABELS).to_csv(
                    os.path.join(a.out, "gold_pred.csv"))
    json.dump({"args": vars(a), "hist": hist, "best": best}, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    print("best", best)


if __name__ == "__main__":
    main()
