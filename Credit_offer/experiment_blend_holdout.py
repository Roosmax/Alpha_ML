import time, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from offer_response_pipeline import (add_features, recency_weights, cat_pool_frame,
                                     ID, TARGET, DATE, CAT_COLS, DROP_FEATS)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
train = add_features(pd.read_csv("train_apps.csv"))
test = add_features(pd.read_csv("test_apps.csv"))
for c in CAT_COLS:
    cats = pd.api.types.union_categoricals([train[c], test[c]]).categories
    train[c] = train[c].cat.set_categories(cats)
feats = [c for c in train.columns if c not in {ID, TARGET, DATE} and c not in DROP_FEATS]
cat_in = [c for c in CAT_COLS if c in feats]
y = train[TARGET].to_numpy().astype(np.int8)
dt = pd.to_datetime(train[DATE]).to_numpy(); order = np.argsort(dt); n = len(train)
i_tr, i_es, i_ev = order[:int(.70*n)], order[int(.70*n):int(.85*n)], order[int(.85*n):]
days = (dt - dt.min()).astype("timedelta64[D]").astype(float)
w = recency_weights(dt, anchor_day=days[i_tr].max())
X = train[feats]; Xc = cat_pool_frame(X)

lp = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
          min_child_samples=120, feature_fraction=0.7, bagging_fraction=0.8,
          bagging_freq=1, lambda_l1=1.0, lambda_l2=2.0, num_threads=0, verbosity=-1)
d = lgb.Dataset(X.iloc[i_tr], y[i_tr], weight=w[i_tr], categorical_feature=cat_in)
v = lgb.Dataset(X.iloc[i_es], y[i_es], reference=d)
bl = lgb.train(lp, d, num_boost_round=8000, valid_sets=[v],
               callbacks=[lgb.early_stopping(150, verbose=False)])
p_lgb = bl.predict(X.iloc[i_ev], num_iteration=bl.best_iteration)
auc_lgb = roc_auc_score(y[i_ev], p_lgb)
log(f"LGB forward AUC = {auc_lgb:.6f} (iter={bl.best_iteration})")

cm = CatBoostClassifier(iterations=6000, learning_rate=0.02, depth=6, l2_leaf_reg=5,
                        loss_function="Logloss", eval_metric="AUC", random_seed=42,
                        allow_writing_files=False, verbose=False, thread_count=-1,
                        early_stopping_rounds=150)
cm.fit(Pool(Xc.iloc[i_tr], y[i_tr], cat_features=cat_in, weight=w[i_tr]),
       eval_set=Pool(Xc.iloc[i_es], y[i_es], cat_features=cat_in))
p_cat = cm.predict_proba(Xc.iloc[i_ev])[:, 1]
auc_cat = roc_auc_score(y[i_ev], p_cat)
log(f"CatBoost forward AUC = {auc_cat:.6f} (iter={cm.get_best_iteration()})")

rk = lambda x: rankdata(x)/len(x)
best = (1.0, auc_lgb)
for wL in np.linspace(0, 1, 21):
    a = roc_auc_score(y[i_ev], wL*rk(p_lgb)+(1-wL)*rk(p_cat))
    if a > best[1]: best = (wL, a)
log(f"Best forward blend: LGB={best[0]:.2f} CAT={1-best[0]:.2f}  AUC={best[1]:.6f}")
log(f" vs LGB-alone {auc_lgb:.6f},  blend gain = {best[1]-auc_lgb:+.6f}")
print("done")
