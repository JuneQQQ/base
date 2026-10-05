#!/usr/bin/env python3
"""Study-level folds.

* The 58 radiologist-labelled studies get fold -1: never trained on by default, scored as a
  sanity check only (58 studies, 9-35 positives per label -> +-0.03 noise per label).
* Studies with byte-identical reports (131 studies in 46 groups) stay in one fold, otherwise
  the same target leaks into train and validation.
* Stratified on a few binarised labels so every fold sees rare findings.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from knee.config import LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--labels", required=True, help="soft label csv from 02_extract_labels.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tr = pd.read_csv(a.train_csv)
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    lab = pd.read_csv(a.labels, index_col=0)
    lab.index = lab.index.astype(str)
    gold = tr[LABELS].notna().all(axis=1)
    df = tr.loc[~gold, ["StudyInstanceUID", "Report"]].copy()
    y = (lab.reindex(df["StudyInstanceUID"])[["ACL", "Medial Meniscus", "Medial OA", "Effusion",
                                               "Fracture"]].fillna(0).values > 0.5).astype(int)
    strat = (y * (2 ** np.arange(y.shape[1]))).sum(1)
    counts = pd.Series(strat).value_counts()
    strat = np.where(pd.Series(strat).map(counts).values < a.n_folds, -1, strat)  # merge rare combos
    groups = df["Report"].fillna("").map(lambda t: hashlib.md5(t.strip().encode()).hexdigest()).values
    skf = StratifiedGroupKFold(n_splits=a.n_folds, shuffle=True, random_state=a.seed)
    df["fold"] = -1
    for k, (_, va) in enumerate(skf.split(df, strat, groups)):
        df.iloc[va, df.columns.get_loc("fold")] = k
    out = pd.concat([df[["StudyInstanceUID", "fold"]],
                     pd.DataFrame({"StudyInstanceUID": tr.loc[gold, "StudyInstanceUID"], "fold": -1})])
    out.to_csv(a.out, index=False)
    print(out["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
