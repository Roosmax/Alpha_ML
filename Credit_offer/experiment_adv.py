import time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from offer_response_pipeline import add_features, ID, TARGET, DATE, CAT_COLS

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

train = add_features(pd.read_csv("train_apps.csv"))
test = add_features(pd.read_csv("test_apps.csv"))
for c in CAT_COLS:
    cats = pd.api.types.union_categoricals([train[c], test[c]]).categories
    train[c] = train[c].cat.set_categories(cats)
    test[c] = test[c].cat.set_categories(cats)
feature_cols = [c for c in train.columns if c not in {ID, TARGET, DATE}]

Xa = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True)
ya = np.r_[np.zeros(len(train)), np.ones(len(test))]
p = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=64,
         min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.8,
         bagging_freq=1, num_threads=0, verbosity=-1)
oof = np.zeros(len(Xa))
skf = StratifiedKFold(5, shuffle=True, random_state=42)
imp = np.zeros(len(feature_cols))
for tr_i, va_i in skf.split(Xa, ya):
    d = lgb.Dataset(Xa.iloc[tr_i], ya[tr_i], categorical_feature=CAT_COLS)
    v = lgb.Dataset(Xa.iloc[va_i], ya[va_i], reference=d)
    b = lgb.train(p, d, num_boost_round=2000, valid_sets=[v],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    oof[va_i] = b.predict(Xa.iloc[va_i], num_iteration=b.best_iteration)
    imp += b.feature_importance("gain") / 5
adv_auc = roc_auc_score(ya, oof)
log(f"Train vs test AUC = {adv_auc:.4f} (0.5=identical, 1.0=fully separable)")
ti = pd.DataFrame({"feature": feature_cols, "gain": imp}).sort_values("gain", ascending=False)
log("Top 12 features:\n" + ti.head(12).to_string(index=False))

# Importance weight for train rows
adv_train = oof[:len(train)]
iw = adv_train / (1 - adv_train + 1e-6)
iw = np.clip(iw / np.median(iw), 0.1, 10.0)

y = train[TARGET].to_numpy().astype(np.int8); X = train[feature_cols]
dt = pd.to_datetime(train[DATE]).to_numpy(); order = np.argsort(dt); n = len(train)
i_tr, i_es, i_ev = order[:int(.70*n)], order[int(.70*n):int(.85*n)], order[int(.85*n):]
days = (dt - dt.min()).astype("timedelta64[D]").astype(float); cut = days[i_tr].max()
wexp = np.clip(np.power(0.5, (cut - days)/120.0), 0.05, 1.0)

base = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
            min_child_samples=120, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, lambda_l1=1.0, lambda_l2=2.0, num_threads=0, verbosity=-1)
def run(name, w):
    d = lgb.Dataset(X.iloc[i_tr], y[i_tr], weight=(w[i_tr] if w is not None else None),
                    categorical_feature=CAT_COLS)
    v = lgb.Dataset(X.iloc[i_es], y[i_es], reference=d)
    b = lgb.train(base, d, num_boost_round=8000, valid_sets=[v],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    log(f"{name:34s} AUC={roc_auc_score(y[i_ev], b.predict(X.iloc[i_ev], num_iteration=b.best_iteration)):.6f}")
run("no weight", None)
run("adversarial importance wt", iw)
run("recency exp120", wexp)
run("recency * adversarial", wexp*iw)
print("done")
