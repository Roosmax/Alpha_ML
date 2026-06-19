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
all_feats = [c for c in train.columns if c not in {ID, TARGET, DATE}]
y = train[TARGET].to_numpy().astype(np.int8)
dt = pd.to_datetime(train[DATE]).to_numpy(); order = np.argsort(dt); n = len(train)
i_tr, i_es, i_ev = order[:int(.70*n)], order[int(.70*n):int(.85*n)], order[int(.85*n):]
days = (dt - dt.min()).astype("timedelta64[D]").astype(float); cut = days[i_tr].max()
w = np.clip(np.power(0.5, (cut - days)/120.0), 0.05, 1.0)
base = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
            min_child_samples=120, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, lambda_l1=1.0, lambda_l2=2.0, num_threads=0, verbosity=-1)

def run(name, feats):
    cats = [c for c in CAT_COLS if c in feats]
    X = train[feats]
    d = lgb.Dataset(X.iloc[i_tr], y[i_tr], weight=w[i_tr], categorical_feature=cats)
    v = lgb.Dataset(X.iloc[i_es], y[i_es], reference=d)
    b = lgb.train(base, d, num_boost_round=8000, valid_sets=[v],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    auc = roc_auc_score(y[i_ev], b.predict(X.iloc[i_ev], num_iteration=b.best_iteration))
    log(f"{name:42s} AUC={auc:.6f} iter={b.best_iteration} nfeat={len(feats)}")

run("all (recency exp120)", all_feats)
run("drop dt_year", [c for c in all_feats if c != "dt_year"])
run("drop dt_year,cb_rate", [c for c in all_feats if c not in {"dt_year","cb_rate"}])
run("drop dt_year,cb_rate,rate_spread", [c for c in all_feats if c not in {"dt_year","cb_rate","rate_spread"}])
run("drop dt_year,cb_rate,offered_rate,rate_spread",
    [c for c in all_feats if c not in {"dt_year","cb_rate","offered_rate","rate_spread"}])
run("drop all dt_*,cb_rate", [c for c in all_feats if not c.startswith("dt_") and c != "cb_rate"])
print("done")
