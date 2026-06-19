"""
Credit Scoring pipeline
Alfa-Bank and MIPT 
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

ID = "id"
RN = "rn"
TARGET = "flag"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alfa credit scoring")
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, default=Path("submission_v2.csv"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recent-k", type=int, default=5,
                   help="Size of the recent-products window for the second bag block.")
    p.add_argument("--learning-rate", type=float, default=0.04)
    p.add_argument("--num-leaves", type=int, default=192)
    p.add_argument("--n-estimators", type=int, default=10000)
    p.add_argument("--early-stopping", type=int, default=150)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: subsample ids, 2 folds, small trees.")
    p.add_argument("--quick-ids", type=int, default=200_000)
    p.add_argument("--threads", type=int, default=0,
                   help="LightGBM num_threads (0 = all cores).")
    p.add_argument("--tag", type=str, default="v2",
                   help="Suffix for saved artifacts.")
    return p.parse_args()

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# Feature engineering
def feature_columns(path: Path) -> list[str]:
    schema = pq.ParquetFile(str(path)).schema_arrow
    return [c for c in schema.names if c not in (ID, RN)]

def column_max(path: Path, col: str) -> int:
    arr = pq.read_table(str(path), columns=[col]).column(0).to_numpy()
    return int(arr.max())

def build_features(
    path: Path,
    cols: list[str],
    cardinalities: dict[str, int],
    recent_k: int,
    keep_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Returns (ids, X, feature_names, categorical_feature_names)."""
    tbl = pq.read_table(str(path), columns=[ID, RN])
    id_arr = tbl.column(0).to_numpy()
    rn_arr = tbl.column(1).to_numpy().astype(np.int16)
    del tbl

    keep_mask = None
    if keep_ids is not None:
        keep_mask = np.isin(id_arr, keep_ids)
        id_arr = id_arr[keep_mask]
        rn_arr = rn_arr[keep_mask]

    uniq_ids, id_codes = np.unique(id_arr, return_inverse=True)
    n_ids = uniq_ids.shape[0]
    n_rows = id_codes.shape[0]
    del id_arr

    n_records = np.bincount(id_codes, minlength=n_ids).astype(np.float32)
    max_rn = np.zeros(n_ids, dtype=np.int16)
    np.maximum.at(max_rn, id_codes, rn_arr)
    recent_mask = rn_arr > (max_rn[id_codes] - recent_k)
    last_mask = rn_arr == max_rn[id_codes]
    last_codes = id_codes[last_mask]
    del rn_arr, max_rn

    paym_cols = [c for c in cols if c.startswith("enc_paym_")]
    pay_card = max(cardinalities[c] for c in paym_cols) if paym_cols else 0

    width = 1 + 2 * sum(cardinalities[c] for c in cols) + 2 * len(cols) + 3 * pay_card
    X = np.zeros((n_ids, width), dtype=np.float32)
    names: list[str] = []
    off = 0

    def put(block: np.ndarray, block_names: list[str]) -> None:
        nonlocal off
        if block.ndim == 1:
            block = block[:, None]
        X[:, off:off + block.shape[1]] = block
        names.extend(block_names)
        off += block.shape[1]

    put(n_records, ["n_records"])

    paym_cnt = np.zeros((n_rows, pay_card), dtype=np.int8) if pay_card else None
    mean_stage = np.zeros((n_ids, len(cols)), dtype=np.float32)
    last_stage = np.zeros((n_ids, len(cols)), dtype=np.float32)

    for j, col in enumerate(cols):
        vals = pq.read_table(str(path), columns=[col]).column(0).to_numpy()
        if keep_mask is not None:
            vals = vals[keep_mask]
        vals = vals.astype(np.int16)
        card = cardinalities[col]

        flat = id_codes * card + vals
        bag = np.bincount(flat, minlength=n_ids * card).reshape(n_ids, card)
        put(bag.astype(np.float32), [f"{col}__{v}" for v in range(card)])
        rbag = np.bincount(flat[recent_mask], minlength=n_ids * card).reshape(n_ids, card)
        put(rbag.astype(np.float32), [f"r{recent_k}_{col}__{v}" for v in range(card)])
        del flat, bag, rbag

        mean_stage[:, j] = (
            np.bincount(id_codes, weights=vals, minlength=n_ids) / n_records
        )
        last_stage[last_codes, j] = vals[last_mask]

        if paym_cnt is not None and col in paym_cols:
            for v in range(min(card, pay_card)):
                paym_cnt[:, v] += (vals == v)
        del vals

    put(mean_stage, [f"mean__{c}" for c in cols])
    del mean_stage
    cat_names = [f"last__{c}" for c in cols]
    put(last_stage, cat_names)
    del last_stage

    if paym_cnt is not None:
        for v in range(pay_card):
            cv = paym_cnt[:, v].astype(np.float32)
            s = np.bincount(id_codes, weights=cv, minlength=n_ids)
            put((s / n_records).astype(np.float32), [f"paym{v}_mean"])
            mx = np.zeros(n_ids, dtype=np.float32)
            np.maximum.at(mx, id_codes, cv)
            put(mx, [f"paym{v}_max"])
            put(s.astype(np.float32), [f"paym{v}_sum"])
            del cv, s, mx
        del paym_cnt

    assert off == width, f"feature width mismatch: {off} != {width}"
    gc.collect()
    return uniq_ids, X, names, cat_names

# Main
def main() -> None:
    args = parse_args()
    d: Path = args.data_dir
    train_path = d / "train_data.parquet"
    test_path = d / "test_data.parquet"

    t0 = time.time()
    log("Loading target / submission ...")
    target = pd.read_csv(d / "train_target.csv")
    sample_sub = pd.read_csv(d / "sample_submission.csv")

    cols = feature_columns(train_path)
    log(f"{len(cols)} source columns")

    log("Computing per-column cardinalities over train+test ...")
    cardinalities = {
        c: max(column_max(train_path, c), column_max(test_path, c)) + 1 for c in cols
    }

    keep_ids = None
    if args.quick:
        rng = np.random.default_rng(args.seed)
        keep_ids = np.sort(
            rng.choice(target[ID].to_numpy(),
                       size=min(args.quick_ids, len(target)), replace=False)
        )
        log(f"QUICK mode: subsampling {len(keep_ids)} train ids")

    log("Building TRAIN features (v2) ...")
    train_ids, Xtr, feat_names, cat_names = build_features(
        train_path, cols, cardinalities, args.recent_k, keep_ids
    )
    log(f"Xtr={Xtr.shape}  ({Xtr.nbytes/1e9:.2f} GB)  cat_feats={len(cat_names)}")

    y = target.set_index(ID)[TARGET].reindex(train_ids).to_numpy()
    assert not np.isnan(y).any(), "Some train ids have no target label"
    y = y.astype(np.int8)
    log(f"y positives={int(y.sum())} / {len(y)} ({y.mean():.4f})")

    log("Building TEST features (v2) ...")
    test_ids, Xte, _, _ = build_features(
        test_path, cols, cardinalities, args.recent_k, None
    )
    log(f"Xte={Xte.shape}  ({Xte.nbytes/1e9:.2f} GB)")

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=100,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        max_bin=255,
        num_threads=args.threads,
        verbosity=-1,
        seed=args.seed,
    )
    n_estimators = args.n_estimators
    early = args.early_stopping
    n_folds = max(2, args.folds)
    if args.quick:
        params.update(learning_rate=0.1, num_leaves=64)
        n_estimators, early, n_folds = 400, 40, 2

    ds = lgb.Dataset(
        Xtr, label=y, feature_name=feat_names,
        categorical_feature=cat_names, free_raw_data=False,
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(Xte.shape[0], dtype=np.float64)
    importances = np.zeros(Xtr.shape[1], dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr, y), 1):
        log(f"Fold {fold}/{n_folds}: train={len(tr_idx)} valid={len(va_idx)}")
        booster = lgb.train(
            params,
            ds.subset(tr_idx),
            num_boost_round=n_estimators,
            valid_sets=[ds.subset(va_idx)],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(early, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof[va_idx] = booster.predict(Xtr[va_idx], num_iteration=booster.best_iteration)
        test_pred += booster.predict(Xte, num_iteration=booster.best_iteration) / n_folds
        importances += booster.feature_importance(importance_type="gain") / n_folds
        log(f"  fold AUC = {roc_auc_score(y[va_idx], oof[va_idx]):.6f}  "
            f"best_iter={booster.best_iteration}")
        del booster
        gc.collect()

    oof_auc = roc_auc_score(y, oof)
    log(f"==== OOF ROC-AUC = {oof_auc:.6f} ====")

    np.save(d / f"oof_{args.tag}.npy", oof)
    np.save(d / f"oof_ids_{args.tag}.npy", train_ids)
    np.save(d / f"test_pred_{args.tag}.npy", test_pred)
    np.save(d / f"test_ids_{args.tag}.npy", test_ids)

    imp = (
        pd.DataFrame({"feature": feat_names, "gain": importances})
        .sort_values("gain", ascending=False)
    )
    imp.to_csv(d / "feature_importance_v2.csv", index=False)
    log("Top 15 features by gain:\n" + imp.head(15).to_string(index=False))

    pred_by_id = pd.Series(test_pred, index=test_ids)
    sub = sample_sub.copy()
    sub[TARGET] = pred_by_id.reindex(sub[ID]).to_numpy()
    missing = sub[TARGET].isna().sum()
    if missing:
        log(f"WARNING: {missing} submission ids missing; filling with mean.")
        sub[TARGET] = sub[TARGET].fillna(test_pred.mean())
    sub.to_csv(args.output, index=False, float_format="%.6f")
    log(f"Saved submission -> {args.output.resolve()}  ({sub.shape[0]} rows)")
    log(f"submission flag stats: min={sub[TARGET].min():.4f} "
        f"mean={sub[TARGET].mean():.4f} max={sub[TARGET].max():.4f}")
    log(f"Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
