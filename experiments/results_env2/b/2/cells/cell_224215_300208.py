import pandas as pd, numpy as np

tt = train_targets()
cv = load_saved("churn_vol_v1.parquet")   # E010 table (best)
m = tt.merge(cv, on=["household_key","snapshot_day"], how="left")
y = m.future_spend_4w
feats = [c for c in cv.columns if c not in ("household_key","snapshot_day")]
corr = {}
for c in feats:
    v = m[c]
    if v.dtype == bool: v = v.astype(float)
    if not np.issubdtype(np.asarray(v).dtype, np.number):
        continue
    corr[c] = np.corrcoef(v.fillna(v.median()), y)[0,1]
cs = pd.Series(corr).sort_values(key=lambda s: s.abs(), ascending=False)
print(cs.head(45))
print("\n--- weakest |corr| ---")
print(cs.tail(15))
