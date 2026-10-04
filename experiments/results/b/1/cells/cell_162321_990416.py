import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, baseline_features

t = train_targets()
print("target describe:")
print(t["future_spend_4w"].describe().round(2))
print(t.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(1))

for n in ["e001_history","e003_dept_mix","e004_long_hist","e005_seasonal_peer","e006_seq_gaps","e007_new","e010_composite","e011_display","e002_marketing"]:
    df = load_saved(n + ".parquet")
    print(n, df.shape, "ncol_feat=", df.shape[1]-2)

# top correlations for e013 (redo, print top 25)
e13 = load_saved("e013_union.parquet")
m = e13.merge(t, on=["household_key","snapshot_day"], how="inner")
y = m["future_spend_4w"].astype(float).values
feats = [c for c in e13.columns if c not in ("household_key","snapshot_day")]
res = []
for f in feats:
    x = pd.to_numeric(m[f], errors="coerce")
    if x.notna().sum() < 50: res.append((f, np.nan)); continue
    c = np.corrcoef(x.fillna(x.median()).values, y)[0,1]
    res.append((f, c))
res.sort(key=lambda r: -(r[1] if r[1]==r[1] else -99))
print("\nTOP 25 |corr|:")
for f,c in res[:25]: print(f"{f:24s} {c:+.4f}")
print("BOTTOM 10:")
for f,c in res[-10:]: print(f"{f:24s} {c:+.4f}")
