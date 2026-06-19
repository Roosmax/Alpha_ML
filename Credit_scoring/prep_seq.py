"""
Prepare credit-history sequences
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold

ID, RN, TARGET = "id", "rn", "flag"
OUT = Path("seq_cache")
OUT.mkdir(exist_ok=True)

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def feature_columns(path: str) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    return [c for c in schema.names if c not in (ID, RN)]

def prep_split(path: str, split: str, cols: list[str]) -> np.ndarray:
    log(f"--- {split}: reading id/rn ---")
    tbl = pq.read_table(path, columns=[ID, RN])
    id_arr = tbl.column(0).to_numpy()
    rn_arr = tbl.column(1).to_numpy().astype(np.int16)
    del tbl

    # sort rows by (id, rn)
    order = np.lexsort((rn_arr, id_arr))
    id_sorted = id_arr[order]
    del rn_arr

    uniq_ids, counts = np.unique(id_sorted, return_counts=True)
    offsets = np.zeros(len(uniq_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    log(f"{split}: {len(id_arr)} rows, {len(uniq_ids)} ids, max len {counts.max()}")

    feats = np.empty((len(id_arr), len(cols)), dtype=np.int8)
    for j, col in enumerate(cols):
        v = pq.read_table(path, columns=[col]).column(0).to_numpy()
        assert v.max() < 128
        feats[:, j] = v[order].astype(np.int8)
    del id_arr, id_sorted

    np.save(OUT / f"{split}_feats.npy", feats)
    np.save(OUT / f"{split}_offsets.npy", offsets)
    np.save(OUT / f"{split}_ids.npy", uniq_ids)
    log(f"{split}: saved feats {feats.shape} ({feats.nbytes/1e9:.2f} GB)")
    return uniq_ids

def main() -> None:
    cols = feature_columns("train_data.parquet")
    log(f"{len(cols)} feature columns")

    cards = np.zeros(len(cols), dtype=np.int16)
    for j, c in enumerate(cols):
        mx = 0
        for p in ("train_data.parquet", "test_data.parquet"):
            mx = max(mx, int(pq.read_table(p, columns=[c]).column(0).to_numpy().max()))
        cards[j] = mx + 1
    np.save(OUT / "cards.npy", cards)
    log(f"cards: min={cards.min()} max={cards.max()} sum={cards.sum()}")

    train_ids = prep_split("train_data.parquet", "train", cols)
    prep_split("test_data.parquet", "test", cols)

    target = pd.read_csv("train_target.csv")
    y = target.set_index(ID)[TARGET].reindex(train_ids).to_numpy()
    assert not np.isnan(y).any()
    y = y.astype(np.int8)
    np.save(OUT / "train_y.npy", y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = np.full(len(y), -1, dtype=np.int8)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[va] = f
    np.save(OUT / "folds.npy", folds)
    log(f"folds saved, sizes: {np.bincount(folds)}")
    log("DONE prep")

if __name__ == "__main__":
    main()
