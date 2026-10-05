import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
print("e007:", df.shape)
print(df.dtypes.value_counts())
print("tt:", tt.shape)
print(tt.future_spend_4w.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print("rows per snapshot:"); print(df.groupby("snapshot_day").size())
m = df.merge(tt, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"]
print("missing target after merge (should be val rows):", int(y.isna().sum()))
feats = [c for c in df.columns if c not in ("household_key","snapshot_day")]
rows = []
for c in feats:
    s = m[c]
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.factorize(pd.Series(s))[0]
    s = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    rows.append((c, round(s.notna().mean(),3), s.corr(y), s.corr(y, method="spearman")))
r = pd.DataFrame(rows, columns=["feat","nonnull","pear","spear"]).set_index("feat")
r["a"] = r.spear.abs()
print(r.sort_values("a", ascending=False).to_string())
