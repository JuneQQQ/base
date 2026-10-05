#!/usr/bin/env python3
"""Soft bootstrapping: blend report labels with OUT-OF-FOLD image-model predictions.

What the forum reported (thread 735304):
  * the image model ("student") beats its report-label "teacher" on the 58 gold studies,
    mostly on findings the reports are silent about (synovitis, effusion, PF OA);
  * replacing labels with predictions, masking noisy cells, or rounding to 0/1 did NOT help;
    MIXING did (50/50 already helped, more weight on the model for silent cells helped more);
  * predictions must be out-of-fold - in-fold predictions just parrot the labels back;
  * never validate on the blended labels (CV improves by construction); keep the original
    report labels as the validation target.

  python scripts/05_pseudo_label.py --labels labels.csv --oof "runs/*/oof_fold*.csv" --out labels_pl.csv
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from knee.config import LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--oof", required=True, help="glob of oof_fold*.csv (several models are averaged)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the report label")
    ap.add_argument("--alpha-silent", type=float, default=0.3,
                    help="weight of the report label where the report was silent (w_<label> < 1)")
    a = ap.parse_args()

    lab = pd.read_csv(a.labels, index_col=0)
    lab.index = lab.index.astype(str)
    files = sorted(glob.glob(a.oof))
    if not files:
        raise SystemExit(f"no files match {a.oof}")
    preds = [pd.read_csv(f, index_col=0) for f in files]
    oof = pd.concat(preds).groupby(level=0).mean()  # a study appears once per model
    oof.index = oof.index.astype(str)
    idx = lab.index.intersection(oof.index)
    print(f"{len(files)} oof files, {len(idx)}/{len(lab)} studies get blended")
    out = lab.copy()
    for c in LABELS:
        w = f"w_{c}"
        silent = lab.loc[idx, w] < 1 if w in lab else pd.Series(False, index=idx)
        alpha = silent.map({True: a.alpha_silent, False: a.alpha})
        out.loc[idx, c] = alpha * lab.loc[idx, c] + (1 - alpha) * oof.loc[idx, c]
        if w in out:
            out.loc[idx, w] = 1.0  # the blended cell is now informed by the image model
    out.to_csv(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
