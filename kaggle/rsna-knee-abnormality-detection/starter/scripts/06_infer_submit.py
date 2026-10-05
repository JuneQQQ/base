#!/usr/bin/env python3
"""Kaggle submission: hidden test DICOMs -> submission.csv (internet off, <= 9 h; aim for minutes).

In the notebook attach (1) the competition, (2) a dataset containing this starter folder
(for the `knee` package), (3) a dataset with checkpoints. Then run:

  !python /kaggle/input/<your-code-dataset>/starter/scripts/06_infer_submit.py \
      --ckpt "/kaggle/input/<your-weights>/*/best.pt"

Design notes from the forum:
  * The 3 visible test studies prove nothing; the hidden run has ~1,300. Before submitting,
    run this script on ~1,300 *training* studies (--split train --limit 1300) to check the
    runtime and that nothing crashes.
  * Use fp16 on T4 (bf16 is emulated there, ~50x slower), no DataParallel (OOMs host RAM on
    the full test), and overlap CPU decoding with GPU work.
  * Never let one bad study kill the run, but log every fallback: a silent 0.5 fallback on
    half the studies looks exactly like a bad model on the LB.
  * The metric is per-label AUC, so ensembles are blended as per-label rank means.
"""
import argparse
import glob
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


def _add_knee_to_path():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if os.path.isdir(os.path.join(here, "knee")):
        sys.path.insert(0, here)
        return
    for d in glob.glob("/kaggle/input/*/") + glob.glob("/kaggle/input/datasets/*/*/"):
        for cand in (d, os.path.join(d, "starter")):
            if os.path.isdir(os.path.join(cand, "knee")):
                sys.path.insert(0, cand)
                return


_add_knee_to_path()
import torch  # noqa: E402

from knee.config import KAGGLE_COMP_ROOTS, LABELS, PrepConfig  # noqa: E402
from knee.dicom_io import build_study  # noqa: E402
from knee.model import load_checkpoint, predict_study, window_index  # noqa: E402

_PREP = None
_SER = None
_DIR = None


def _init(prep, ser, d):
    global _PREP, _SER, _DIR
    _PREP, _SER, _DIR = prep, ser, d


def _build(uid):
    try:
        vol, mask, _ = build_study(os.path.join(_DIR, uid), _SER.get(uid, []), _PREP)
        return uid, vol, mask, None
    except Exception as e:
        return uid, None, None, f"{type(e).__name__}: {e}"


def prep_from_ckpt(ck) -> PrepConfig:
    d = dict(ck["prep"])
    d["slots"] = tuple(tuple(s) for s in d["slots"])
    d["span"] = tuple(d["span"])
    return PrepConfig(**d)


def rank_pct(p):
    r = pd.DataFrame(p).rank(axis=0, method="average").values
    return (r - 1) / max(1, len(p) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="checkpoint paths or globs")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--split", default="test", choices=["test", "train"], help="train = dry run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/kaggle/working/submission.csv")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--time-budget-h", type=float, default=8.3)
    a = ap.parse_args()
    t0 = time.time()

    root = a.data_dir or next((r for r in KAGGLE_COMP_ROOTS if os.path.exists(os.path.join(r, "test.csv"))), None)
    if root is None:
        raise SystemExit("competition data not found; pass --data-dir")
    ids = pd.read_csv(os.path.join(root, f"{a.split}.csv"), usecols=["StudyInstanceUID"])["StudyInstanceUID"].astype(str).tolist()
    if a.limit:
        ids = ids[:a.limit]
    ser = pd.read_csv(os.path.join(root, f"{a.split}_series.csv"), dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    sub_cols = ["StudyInstanceUID"] + LABELS
    ss = os.path.join(root, "sample_submission.csv")
    if os.path.exists(ss):
        sub_cols = list(pd.read_csv(ss, nrows=1).columns)

    paths = sorted({p for g in a.ckpt for p in glob.glob(g, recursive=True)})
    if not paths:
        raise SystemExit(f"no checkpoints match {a.ckpt}")
    n_gpu = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(n_gpu)] or [torch.device("cpu")]
    # one copy of every model per GPU; each GPU serves its own share of the studies
    models = {d: [] for d in devices}
    prep = None
    for p in paths:
        for d in devices:
            m, ck = load_checkpoint(p, d)
            models[d].append((m, ck["img"]))
        this = prep_from_ckpt(ck)
        if prep is not None and this != prep:
            raise SystemExit("all checkpoints must share one preprocessing config (group them per run)")
        prep = this
    centres, slot_ids = window_index(prep)
    print(f"{len(ids)} studies | {len(paths)} checkpoints | devices {devices} | {time.time()-t0:.0f}s", flush=True)

    N, M = len(ids), len(paths)
    probs = np.full((M, N, len(LABELS)), np.nan, np.float32)
    pos = {u: i for i, u in enumerate(ids)}
    fails, q = [], queue.Queue(maxsize=8 * len(devices))
    deadline = t0 + a.time_budget_h * 3600

    def gpu_worker(dev):
        while True:
            item = q.get()
            if item is None:
                return
            uid, vol, mask = item
            for j, (m, img) in enumerate(models[dev]):
                try:
                    probs[j, pos[uid]] = predict_study(m, vol, mask, img, centres, slot_ids, dev,
                                                       dtype=torch.float16, tta_shift=a.tta)
                except Exception as e:
                    fails.append((uid, f"model {j}: {type(e).__name__}: {e}"))

    threads = [threading.Thread(target=gpu_worker, args=(d,), daemon=True) for d in devices]
    for t in threads:
        t.start()
    # decode in chunks so finished-but-unconsumed stacks can't pile up in host RAM and a
    # time-budget stop doesn't wait for the whole remaining test set
    chunk, k = 8 * a.workers, 0
    with ProcessPoolExecutor(a.workers, initializer=_init,
                             initargs=(prep, SER, os.path.join(root, f"{a.split}_series"))) as ex:
        for s in range(0, N, chunk):
            if time.time() > deadline:
                fails.append(("*", f"time budget hit after {k} studies; the rest get column-mean fallbacks"))
                break
            for uid, vol, mask, err in ex.map(_build, ids[s:s + chunk]):
                k += 1
                if err is not None:
                    fails.append((uid, err))
                else:
                    q.put((uid, vol, mask))
                if k % 100 == 0 or k == N:
                    el = time.time() - t0
                    print(f"  {k}/{N} decoded | {el/60:.1f} min | eta {el/k*(N-k)/60:.1f} min", flush=True)
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()

    # per-model fill (column mean) -> per-label rank -> mean over models
    for j in range(M):
        col_mean = np.nanmean(probs[j], axis=0)
        bad = np.isnan(probs[j])
        probs[j][bad] = np.take(np.nan_to_num(col_mean, nan=0.5), np.nonzero(bad)[1])
    final = np.mean([rank_pct(probs[j]) for j in range(M)], axis=0) if N > 1 else np.nanmean(probs, 0)

    sub = pd.DataFrame(final.astype(np.float32), columns=LABELS)
    sub.insert(0, "StudyInstanceUID", ids)
    sub = sub[sub_cols]
    assert sub["StudyInstanceUID"].tolist() == ids and np.isfinite(sub[LABELS].values).all()
    sub.to_csv(a.out, index=False)
    print(f"fallbacks: {len(fails)}", *fails[:20], sep="\n  ")
    print(f"wrote {a.out} {sub.shape} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
