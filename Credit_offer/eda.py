"""EDA the credit offer """
import pandas as pd
import numpy as np

OUT = []
def log(*a): OUT.append(" ".join(str(x) for x in a))

train = pd.read_csv("train_apps.csv")
test = pd.read_csv("test_apps.csv")
sub = pd.read_csv("sample_submission_new.csv")

log("shapes")
log("train:", train.shape, "| test:", test.shape, "| sample_sub:", sub.shape)

log("sample_submission_new columns")
log("columns:", list(sub.columns))
log(str(sub.head()))
log("dtypes:\n" + str(sub.dtypes))

log("train columns")
log("columns:", list(train.columns))
log("train dtypes")
log(str(train.dtypes))

# identify target
train_only = [c for c in train.columns if c not in test.columns]
test_only = [c for c in test.columns if c not in train.columns]
log("columns in train not in test:", train_only)
log("columns in test not in train:", test_only)
log("")

# Id column candidates
log("sample_sub vs train shared cols:", [c for c in sub.columns if c in train.columns])
log("")

# Target analysis
for tcol in train_only:
    s = train[tcol]
    log(f"target '{tcol}'")
    log("dtype:", s.dtype, "| nunique:", s.nunique())
    log("value_counts:\n" + str(s.value_counts(dropna=False).head(20)))
    if pd.api.types.is_numeric_dtype(s):
        log("mean:", s.mean())
    log("")

log("per column summary")
for c in train.columns:
    s = train[c]
    na = s.isna().mean()
    nun = s.nunique(dropna=True)
    dt = s.dtype
    extra = ""
    if pd.api.types.is_numeric_dtype(s):
        extra = f"min={s.min()} max={s.max()} mean={round(float(s.mean()),3) if s.notna().any() else 'NA'}"
    else:
        top = s.value_counts(dropna=True).head(5).index.tolist()
        extra = f"top={top}"
    log(f"   {c}: dtype={dt} na={round(na,3)} nunique={nun} {extra}")
log("")

log("Train, first 6 rows, transposed")
log(str(train.head(6).T))

with open("eda_apps_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("train", train.shape, "test", test.shape)
