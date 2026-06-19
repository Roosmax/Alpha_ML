""" EDA for the dataset. """
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

OUT = []
def log(*a):
    OUT.append(" ".join(str(x) for x in a))

# Parquet metadata
for fname in ["train_data.parquet", "test_data.parquet"]:
    pf = pq.ParquetFile(fname)
    md = pf.metadata
    log(f"===== {fname} =====")
    log("num_rows:", md.num_rows)
    log("num_row_groups:", md.num_row_groups)
    log("num_columns:", md.num_columns)
    schema = pf.schema_arrow
    log("columns:", schema.names)
    log("dtypes:")
    for n in schema.names:
        log("   ", n, "->", schema.field(n).type)
    log("")

# Target and submission
tgt = pd.read_csv("train_target.csv")
sub = pd.read_csv("sample_submission.csv")
log("train_target.csv")
log("shape:", tgt.shape)
log("columns:", list(tgt.columns))
log("dtypes:\n" + str(tgt.dtypes))
log("flag value_counts:\n" + str(tgt["flag"].value_counts(dropna=False)))
log("flag mean (default rate):", tgt["flag"].mean())
log("n unique id:", tgt["id"].nunique())
log("id min/max:", tgt["id"].min(), tgt["id"].max())
log("head:\n" + str(tgt.head()))
log("")
log("sample_submission.csv")
log("shape:", sub.shape)
log("columns:", list(sub.columns))
log("dtypes:\n" + str(sub.dtypes))
log("n unique id:", sub["id"].nunique())
log("id min/max:", sub["id"].min(), sub["id"].max())
log("head:\n" + str(sub.head()))
log("")

# Read first row group only
pf = pq.ParquetFile("train_data.parquet")
sample = pf.read_row_group(0).to_pandas()
log("train_data sample")
log("sample shape:", sample.shape)
log("dtypes:\n" + str(sample.dtypes))
log("head:\n" + str(sample.head(8).to_string()))
log("")

# Rows per id without first row group
rpi = sample.groupby("id").size()
log("rows per id: min/median/mean/max =",
    rpi.min(), rpi.median(), rpi.mean(), rpi.max())
log("")

# cardinality and range per column on the sample
log("per-column stats")
for c in sample.columns:
    s = sample[c]
    nun = s.nunique(dropna=True)
    na = s.isna().sum()
    try:
        mn, mx = s.min(), s.max()
    except Exception:
        mn, mx = "NA", "NA"
    log(f"   {c}: nunique={nun} na={na} min={mn} max={mx} dtype={s.dtype}")
log("")

# Value counts for 15 top
for c in [col for col in ["rn", "enc_loans_credit_status", "enc_loans_credit_type",
                          "pre_loans5", "pre_util", "is_zero_util"] if col in sample.columns]:
    log(f"value_counts {c} (top 15):\n" + str(sample[c].value_counts(dropna=False).head(15)))
    log("")

# Id overlap check
train_ids_sample = set(sample["id"].unique())
test_pf = pq.ParquetFile("test_data.parquet")
test_sample = test_pf.read_row_group(0).to_pandas()
test_ids_sample = set(test_sample["id"].unique())
log("train id sample range:", min(train_ids_sample), max(train_ids_sample))
log("test id sample range:", min(test_ids_sample), max(test_ids_sample))
log("overlap in samples:", len(train_ids_sample & test_ids_sample))

with open("eda_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("EDA done;  train rows:", pq.ParquetFile('train_data.parquet').metadata.num_rows)
