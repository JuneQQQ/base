#!/usr/bin/env python3
"""Decode every study once into a fixed (S, H, W) uint8 stack: <out>/<StudyInstanceUID>.npy.

The raw DICOMs are ~570 GB; after slot selection + 140 mm crop + resize the training input is
~15 GB at 256 px. Decoding is >70% of a naive training run, so do it once (on Kaggle CPU
sessions, which don't use GPU quota) and publish the folder as a private Kaggle dataset.

  python scripts/01_build_cache.py --data-dir /kaggle/input/competitions/rsna-knee-abnormality-detection \
      --split train --out /kaggle/working/cache_256 --workers 4
Resumable: studies whose .npy already exists are skipped.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from knee.config import DEFAULT_PREP, KAGGLE_COMP_ROOTS  # noqa: E402
from knee.dicom_io import build_study  # noqa: E402

PREP = DEFAULT_PREP
SERIES = {}
SERIES_DIR = ""
OUT = ""


def _init(prep, series, series_dir, out):
    global PREP, SERIES, SERIES_DIR, OUT
    PREP, SERIES, SERIES_DIR, OUT = prep, series, series_dir, out


def _work(uid):
    t = time.time()
    try:
        vol, mask, meta = build_study(os.path.join(SERIES_DIR, uid), SERIES.get(uid, []), PREP)
    except Exception as e:  # never lose the whole run to one study; no .npy -> a rerun retries it
        return {"StudyInstanceUID": uid, "side": None, "mask": "", "n_valid": 0,
                "error": f"{type(e).__name__}: {e}", "sec": round(time.time() - t, 2)}
    np.save(os.path.join(OUT, f"{uid}.npy"), vol)
    return {"StudyInstanceUID": uid, "side": meta.get("side"),
            "mask": "".join(map(str, mask.tolist())), "n_valid": int(mask.sum()),
            "error": meta.get("error", ""), "sec": round(time.time() - t, 2)}


def find_root(arg):
    if arg:
        return arg
    for r in KAGGLE_COMP_ROOTS:
        if os.path.exists(os.path.join(r, "train.csv")):
            return r
    raise SystemExit("pass --data-dir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--img-size", type=int, default=DEFAULT_PREP.img_size)
    ap.add_argument("--crop-mm", type=float, default=DEFAULT_PREP.crop_mm)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th study starting at i "
                    "(run n Kaggle sessions in parallel, then merge the folders)")
    a = ap.parse_args()

    root = find_root(a.data_dir)
    prep = replace(DEFAULT_PREP, img_size=a.img_size, crop_mm=a.crop_mm)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "prep.json"), "w") as f:
        json.dump(asdict(prep), f, indent=1)

    studies = pd.read_csv(os.path.join(root, f"{a.split}.csv"), usecols=["StudyInstanceUID"])
    ser = pd.read_csv(os.path.join(root, f"{a.split}_series.csv"), dtype={"StudyInstanceUID": str,
                                                                           "SeriesInstanceUID": str})
    series = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    uids = studies["StudyInstanceUID"].astype(str).tolist()
    i, n = map(int, a.shard.split("/"))
    uids = uids[i::n]
    todo = [u for u in uids if not os.path.exists(os.path.join(a.out, f"{u}.npy"))]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(uids)} studies, {len(todo)} to build, prep={asdict(prep)}", flush=True)

    meta_path = os.path.join(a.out, f"meta_{a.split}_{i}of{n}.csv")
    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init,
              initargs=(prep, series, os.path.join(root, f"{a.split}_series"), a.out)) as pool:
        for k, r in enumerate(pool.imap_unordered(_work, todo, chunksize=2), 1):
            rows.append(r)
            if k % 50 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"{k}/{len(todo)}  {el/60:.1f} min  eta {el/k*(len(todo)-k)/60:.1f} min", flush=True)
                pd.DataFrame(rows).to_csv(meta_path, mode="a", header=not os.path.exists(meta_path), index=False)
                rows = []
    print("done", flush=True)


if __name__ == "__main__":
    main()
