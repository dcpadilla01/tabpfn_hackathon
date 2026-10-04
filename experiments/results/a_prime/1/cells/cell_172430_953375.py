import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("train rows", tr.shape)
y = tr.future_spend_4w.values.astype(float)
num = [c for c in tr.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(tr[c])]
print("n numeric cols", len(num))
cor = {}
for c in num:
    x = tr[c].values.astype(float)
    ok = np.isfinite(x)
    if ok.sum() < 1000: continue
    cor[c] = np.corrcoef(x[ok], y[ok])[0,1]
cor = pd.Series(cor).sort_values(key=np.abs, ascending=False)
print(cor.head(35))
print()
print(cor.tail(10))