import pandas as pd, numpy as np
OUT=[]
def log(*a): OUT.append(" ".join(str(x) for x in a))

train = pd.read_csv("train_apps.csv")
test = pd.read_csv("test_apps.csv")
sub = pd.read_csv("sample_submission_new.csv")

log("train rows:", len(train), "unique front_id:", train.front_id.nunique())
log("test rows:", len(test), "unique front_id:", test.front_id.nunique())
log("sub rows:", len(sub), "unique front_id:", sub.front_id.nunique())
tset, subset = set(test.front_id), set(sub.front_id)
log("testâˆ©sub:", len(tset & subset))
log("in sub not in test:", len(subset - tset))
log("in test not in sub:", len(tset - subset))

log("test front_id duplicated rows:", int(test.front_id.duplicated().sum()))
log("train front_id duplicated rows:", int(train.front_id.duplicated().sum()))

# dates
for name, df in [("train", train), ("test", test)]:
    d = pd.to_datetime(df["decision_day"], errors="coerce")
    log(f"{name} decision_day: min={d.min()} max={d.max()} n_na={int(d.isna().sum())}")

tr = train.copy()
tr["d"] = pd.to_datetime(tr["decision_day"], errors="coerce")
tr["month"] = tr["d"].dt.to_period("M").astype(str)
log("target_value mean by month (train):")
log(str(tr.groupby("month")["target_value"].agg(["mean","count"])))

# Ñategorical target rates
for c in ["db_group_last","cb_rate"]:
    log(f"target mean by {c}:")
    log(str(train.groupby(c, dropna=False)["target_value"].agg(["mean","count"]).sort_values("count",ascending=False).head(12)))

with open("eda_apps2_report.txt","w",encoding="utf-8") as f: f.write("\n".join(OUT))
print("done")
