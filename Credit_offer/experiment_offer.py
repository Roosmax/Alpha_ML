"""Optimize against the HONEST time-holdout (proxy for the future test).
70% oldest -> train, next 15% -> early-stop valid, newest 15% -> eval."""
import time, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from offer_response_pipeline import add_features, ID, TARGET, DATE, CAT_COLS

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

train = add_features(pd.read_csv("train_apps.csv"))
test = add_features(pd.read_csv("test_apps.csv"))
for c in CAT_COLS:
    cats = pd.api.types.union_categoricals([train[c], test[c]]).categories
    train[c] = train[c].cat.set_categories(cats)
feature_cols = [c for c in train.columns if c not in {ID, TARGET, DATE}]
y = train[TARGET].to_numpy().astype(np.int8)
X = train[feature_cols]
dt = pd.to_datetime(train[DATE]).to_numpy()
order = np.argsort(dt)
n = len(train)
i_tr, i_es, i_ev = order[:int(.70*n)], order[int(.70*n):int(.85*n)], order[int(.85*n):]
days = (dt - dt.min()).astype("timedelta64[D]").astype(float)

def run(name, params, w=None):
    dtr = lgb.Dataset(X.iloc[i_tr], y[i_tr], weight=(w[i_tr] if w is not None else None),
                      categorical_feature=CAT_COLS)
    des = lgb.Dataset(X.iloc[i_es], y[i_es], reference=dtr)
    b = lgb.train(params, dtr, num_boost_round=8000, valid_sets=[des],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    auc = roc_auc_score(y[i_ev], b.predict(X.iloc[i_ev], num_iteration=b.best_iteration))
    log(f"{name:38s} AUC={auc:.6f} iter={b.best_iteration}")
    return auc

base = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
            min_child_samples=120, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, lambda_l1=1.0, lambda_l2=2.0, num_threads=0, verbosity=-1)

log("=== baseline & variants (optimizing the honest forward metric) ===")
run("baseline", base)

# linear recency weight (oldest .3 -> newest 1.0)
wlin = 0.3 + 0.7 * (days / days.max())
run("recency linear 0.3..1.0", base, wlin)

# exponential recency, half-life 120 days from the train cutoff
cut = days[i_tr].max()
wexp = np.power(0.5, (cut - days) / 120.0); wexp = np.clip(wexp, 0.05, 1.0)
run("recency exp halflife=120d", base, wexp)

wexp2 = np.power(0.5, (cut - days) / 200.0); wexp2 = np.clip(wexp2, 0.05, 1.0)
run("recency exp halflife=200d", base, wexp2)

run("leaves=96", dict(base, num_leaves=96))
run("leaves=32 mcs=200", dict(base, num_leaves=32, min_child_samples=200))
run("lr=0.01", dict(base, learning_rate=0.01))
run("lr=0.01 + recency exp200", dict(base, learning_rate=0.01), wexp2)
print("done")
