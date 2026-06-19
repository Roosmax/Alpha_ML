"""
Blend all model predictions
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

CACHE = Path("seq_cache")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["v2", "gru"],
                   help="'v2' = LightGBM")
    p.add_argument("--output", type=Path, default=Path("submission_blend.csv"))
    p.add_argument("--step", type=float, default=0.05)
    return p.parse_args()

def load_model(name: str, train_ids: np.ndarray):
    if Path(f"oof_ids_{name}.npy").exists():
        oof = np.load(f"oof_{name}.npy")
        oof_ids = np.load(f"oof_ids_{name}.npy")
        assert np.array_equal(oof_ids, train_ids), f"id order mismatch for {name}"
        return oof, np.load(f"test_pred_{name}.npy")

    oof = np.load(f"oof_{name}.npy")
    mask = np.load(f"oof_{name}_mask.npy")
    assert mask.all(), f"{name}: OOF does not cover all folds yet"
    return oof, np.load(f"test_pred_{name}.npy")

def to_rank(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x)

def to_rank_per_fold(x: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for f in np.unique(folds):
        m = folds == f
        out[m] = rankdata(x[m]) / m.sum()
    return out

def main() -> None:
    args = parse_args()
    train_ids = np.load(CACHE / "train_ids.npy")
    y = np.load(CACHE / "train_y.npy")
    folds = np.load(CACHE / "folds.npy")

    names, oofs, tests = [], [], []
    for group in args.models:
        members = group.split("+")
        g_oof = np.zeros(len(y), dtype=np.float64)
        g_test = None
        for name in members:
            oof, test = load_model(name, train_ids)
            auc_pooled = roc_auc_score(y, oof)
            oof_r = to_rank_per_fold(oof, folds)
            print(f"  {name}: OOF AUC pooled = {auc_pooled:.6f}  "
                  f"per-fold-rank = {roc_auc_score(y, oof_r):.6f}")
            g_oof += oof_r / len(members)
            t = to_rank(test)
            g_test = t / len(members) if g_test is None else g_test + t / len(members)
        if len(members) > 1:
            print(f"{group}: group OOF AUC = {roc_auc_score(y, g_oof):.6f}")
        names.append(group)
        oofs.append(g_oof)
        tests.append(g_test)

    n = len(names)
    grid = np.arange(0.0, 1.0 + 1e-9, args.step)
    best_auc, best_w = -1.0, None
    for w in product(grid, repeat=n - 1):
        w = np.array(list(w) + [1.0 - sum(w)])
        if w[-1] < -1e-9:
            continue
        blend = sum(wi * o for wi, o in zip(w, oofs))
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"Best result = {best_auc:.6f}  weights = "
          + ", ".join(f"{nm}:{wi:.2f}" for nm, wi in zip(names, best_w)))

    test_blend = sum(wi * t for wi, t in zip(best_w, tests))

    test_ids = np.load("test_ids_v2.npy")
    sample = pd.read_csv("sample_submission.csv")
    pred = pd.Series(test_blend, index=test_ids).reindex(sample["id"]).to_numpy()
    assert not np.isnan(pred).any()
    sub = sample.copy()
    sub["flag"] = pred
    sub.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Saved {args.output} ({len(sub)} rows)")

if __name__ == "__main__":
    main()
