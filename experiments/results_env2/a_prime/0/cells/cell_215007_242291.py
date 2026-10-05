import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
print("shape", df.shape)
cols = list(df.columns)
print("n cols", len(cols))
for i in range(0, len(cols), 6):
    print(" | ".join(cols[i:i+6]))

tt = A.train_targets()
y = tt["future_spend_4w"].astype(float)
print("\ntarget describe:\n", y.describe())
print("zero share:", float((y==0).mean()))
print("snapshot_days:", A.snapshot_days())

m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
print("merged shape", m.shape)
ymed = float(y.median())
print("const median MAE:", round(float(np.abs(y-ymed).mean()),2), "median:", ymed)
print("const mean MAE:", round(float(np.abs(y-y.mean()).mean()),2))

feat_cols = [c for c in cols if c not in ("household_key","snapshot_day")]
spend_cols = [c for c in feat_cols if "spend" in c.lower()]
print("\nspend cols:", spend_cols)
for c in spend_cols:
    p = pd.to_numeric(m[c], errors="coerce").fillna(0).values
    if p.std()>0:
        print(f"{c}: rawMAE={np.abs(y-p).mean():.2f} corr={np.corrcoef(y,p)[0,1]:.3f} logcorr={np.corrcoef(y,np.log1p(p))[0,1]:.3f}")

c28 = [c for c in spend_cols if "28" in c and "share" not in c]
if c28:
    p = pd.to_numeric(m[c28[0]], errors="coerce").fillna(0).values
    al = np.linspace(0,2,41); maes=[np.abs(y-a*p).mean() for a in al]; i=int(np.argmin(maes))
    print("\nbest scalar on", c28[0], "alpha", round(float(al[i]),2), "MAE", round(float(maes[i]),2))

num = m[feat_cols].select_dtypes(include=[np.number])
corr = num.corrwith(y).dropna()
top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(30)
print("\ntop |corr| with target:\n", top)
