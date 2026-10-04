import pandas as pd, numpy as np
from agent_api import load_saved, train_targets

for n in ["e013_union","cand_new","e009_macro","e008_decomp2","e012_full","e001_history"]:
    try:
        df = load_saved(n + ".parquet")
        print("==", n, df.shape)
        print(list(df.columns))
        print()
    except Exception as e:
        print("==", n, "ERR", repr(e))

t = train_targets()
print("target rows:", t.shape)
print(t.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median"]).round(2))
print(t["future_spend_4w"].describe().round(2))

e13 = load_saved("e013_union.parquet")
m = e13.merge(t, on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape)
y = m["future_spend_4w"].astype(float).values
feats = [c for c in e13.columns if c not in ("household_key","snapshot_day")]
res = []
for f in feats:
    x = pd.to_numeric(m[f], errors="coerce")
    nu = m[f].nunique()
    if x.notna().sum() < 50 or x.fillna(0).std() == 0:
        res.append((f, np.nan, nu)); continue
    med = x.median()
    c = np.corrcoef(x.fillna(med).values, y)[0,1]
    res.append((f, c, nu))
res.sort(key=lambda r: -(r[1] if r[1]==r[1] else -99))
print("\n--- train corr with target (e013_union features) ---")
for f,c,nu in res:
    print(f"{f:42s} {c:+.4f} nu={nu}")
