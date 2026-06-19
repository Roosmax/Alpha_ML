"""
Credit offer pipeline
Alpha and MIPT
"""
from __future__ import annotations
import argparse, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

ID, TARGET, DATE = "front_id", "target_value", "decision_day"
CAT_COLS = ["db_group_last", "fl_adminarea"]
DROP_FEATS = {"dt_year"}
HALFLIFE_DAYS = 120.0

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 7, 2026])
    p.add_argument("--output", default="submission_offer.csv")
    return p.parse_args()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df[DATE], errors="coerce")
    df["dt_month"] = d.dt.month
    df["dt_dow"] = d.dt.dayofweek
    df["dt_dom"] = d.dt.day
    df["dt_week"] = d.dt.isocalendar().week.astype("float")
    df["dt_year"] = d.dt.year
    df["dt_is_month_end"] = d.dt.is_month_end.astype("int8")
    base = [c for c in df.columns if c not in (ID, TARGET, DATE) and c not in CAT_COLS
            and not c.startswith("dt_")]
    df["n_missing"] = df[base].isna().sum(axis=1)
    if {"offered_rate", "cb_rate"}.issubset(df.columns):
        df["rate_spread"] = df["offered_rate"] - df["cb_rate"]
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
    return df

def recency_weights(dt_values: np.ndarray, anchor_day: float) -> np.ndarray:
    days = (dt_values - dt_values.min()).astype("timedelta64[D]").astype(float)
    w = np.power(0.5, (anchor_day - days) / HALFLIFE_DAYS)
    return np.clip(w, 0.05, 1.0)

def cat_pool_frame(X: pd.DataFrame) -> pd.DataFrame:
    # CatBoost
    X = X.copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("object").where(X[c].notna(), "NA").astype(str)
    return X


def main():
    args = parse_args()
    t0 = time.time()
    log("Load")
    train = add_features(pd.read_csv("train_apps.csv"))
    test = add_features(pd.read_csv("test_apps.csv"))
    sub = pd.read_csv("sample_submission_new.csv")
    for c in CAT_COLS:
        cats = pd.api.types.union_categoricals([train[c], test[c]]).categories
        train[c] = train[c].cat.set_categories(cats)
        test[c] = test[c].cat.set_categories(cats)

    feature_cols = [c for c in train.columns
                    if c not in {ID, TARGET, DATE} and c not in DROP_FEATS]
    cat_in = [c for c in CAT_COLS if c in feature_cols]
    log(f"{len(feature_cols)} features; cat={cat_in}")

    y = train[TARGET].to_numpy().astype(np.int8)
    Xtr, Xte = train[feature_cols], test[feature_cols]
    dt = pd.to_datetime(train[DATE]).to_numpy()
    days_all = (dt - dt.min()).astype("timedelta64[D]").astype(float)
    w_all = recency_weights(dt, anchor_day=days_all.max())

    lgb_params = dict(objective="binary", metric="auc", learning_rate=0.02,
                      num_leaves=64, min_child_samples=120, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l1=1.0,
                      lambda_l2=2.0, num_threads=0, verbosity=-1)

    Xtr_c, Xte_c = cat_pool_frame(Xtr), cat_pool_frame(Xte)
    cat_params = dict(learning_rate=0.02, depth=6, l2_leaf_reg=5,
                      loss_function="Logloss", eval_metric="AUC", random_seed=42,
                      allow_writing_files=False, verbose=False, thread_count=-1)
    rk = lambda x: rankdata(x) / len(x)

    # oldest 70% - train, next 15% - early-stop, newest 15% - eval
    order = np.argsort(dt); n = len(train)
    i_tr, i_es, i_ev = order[:int(.70*n)], order[int(.70*n):int(.85*n)], order[int(.85*n):]

    d_h = lgb.Dataset(Xtr.iloc[i_tr], y[i_tr], weight=w_all[i_tr], categorical_feature=cat_in)
    v_h = lgb.Dataset(Xtr.iloc[i_es], y[i_es], reference=d_h)
    bh = lgb.train(lgb_params, d_h, num_boost_round=8000, valid_sets=[v_h],
                   callbacks=[lgb.early_stopping(150, verbose=False)])
    p_lgb_ev = bh.predict(Xtr.iloc[i_ev], num_iteration=bh.best_iteration)
    R_lgb = int(bh.best_iteration * 1.15)

    ch = CatBoostClassifier(iterations=8000, early_stopping_rounds=150, **cat_params)
    ch.fit(Pool(Xtr_c.iloc[i_tr], y[i_tr], cat_features=cat_in, weight=w_all[i_tr]),
           eval_set=Pool(Xtr_c.iloc[i_es], y[i_es], cat_features=cat_in))
    p_cat_ev = ch.predict_proba(Xtr_c.iloc[i_ev])[:, 1]
    R_cat = int(ch.get_best_iteration() * 1.15)

    auc_lgb_h = roc_auc_score(y[i_ev], p_lgb_ev)
    auc_cat_h = roc_auc_score(y[i_ev], p_cat_ev)
    best_w, best_auc = 1.0, auc_lgb_h
    for wL in np.linspace(0, 1, 21):
        a = roc_auc_score(y[i_ev], wL*rk(p_lgb_ev) + (1-wL)*rk(p_cat_ev))
        if a > best_auc: best_auc, best_w = a, wL
    log(f"** TIME-HOLDOUT (honest LB proxy): LGB={auc_lgb_h:.6f} CAT={auc_cat_h:.6f} "
        f"BLEND={best_auc:.6f} @ LGB={best_w:.2f}/CAT={1-best_w:.2f} **")
    log(f"   rounds: R_lgb={R_lgb}  R_cat={R_cat}")

    # LightGBM
    oof_lgb = np.zeros(n); test_lgb = np.zeros(len(test))
    n_runs = args.folds * len(args.seeds)
    for seed in args.seeds:
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(Xtr, y):
            d = lgb.Dataset(Xtr.iloc[tr_i], y[tr_i], weight=w_all[tr_i], categorical_feature=cat_in)
            b = lgb.train(dict(lgb_params, seed=seed), d, num_boost_round=R_lgb)
            oof_lgb[va_i] += b.predict(Xtr.iloc[va_i]) / len(args.seeds)
            test_lgb += b.predict(Xte) / n_runs

    # CatBoost
    oof_cat = np.zeros(n); test_cat = np.zeros(len(test))
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=42)
    for tr_i, va_i in skf.split(Xtr_c, y):
        model = CatBoostClassifier(iterations=R_cat, **cat_params)
        model.fit(Pool(Xtr_c.iloc[tr_i], y[tr_i], cat_features=cat_in, weight=w_all[tr_i]))
        oof_cat[va_i] = model.predict_proba(Xtr_c.iloc[va_i])[:, 1]
        test_cat += model.predict_proba(Xte_c)[:, 1] / args.folds

    # Blend
    oof = best_w*rk(oof_lgb) + (1-best_w)*rk(oof_cat)
    test_pred = best_w*rk(test_lgb) + (1-best_w)*rk(test_cat)
    np.save("oof_lgb_offer.npy", oof_lgb); np.save("oof_cat_offer.npy", oof_cat)
    np.save("test_lgb_offer.npy", test_lgb); np.save("test_cat_offer.npy", test_cat)

    np.save("oof_offer.npy", oof); np.save("oof_offer_ids.npy", train[ID].to_numpy())
    np.save("test_pred_offer.npy", test_pred); np.save("test_pred_offer_ids.npy", test[ID].to_numpy())

    pred_map = pd.Series(test_pred, index=test[ID].to_numpy())
    oof_map = pd.Series(oof, index=train[ID].to_numpy())
    out = sub[[ID]].copy()
    out[TARGET] = out[ID].map(pred_map)
    miss = out[TARGET].isna()
    out.loc[miss, TARGET] = out.loc[miss, ID].map(oof_map)
    still = int(out[TARGET].isna().sum())
    if still:
        out[TARGET] = out[TARGET].fillna(float(np.median(test_pred)))
    out.to_csv(args.output, index=False, float_format="%.6f")
    log(f"Saved {args.output}: {len(out)} rows ({int((~miss).sum())} test, {int(miss.sum())} OOF, {still} filled)")
    log(f"pred range [{out[TARGET].min():.4f}, {out[TARGET].max():.4f}]  Total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
