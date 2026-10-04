import agent_api as A
import pandas as pd, numpy as np

e011 = A.load_saved("e011_price.parquet")
base = list(A.load_saved("rfm28.parquet").columns)
e011_feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
print("E011 n features:", len(e011_feats))

tt = A.train_targets()
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", df.shape)

y = df["future_spend_4w"].values
ly = np.log1p(y)

num = df[e011_feats].select_dtypes(include=[np.number,bool]).columns
corr = {}
for c in num:
    x = pd.to_numeric(df[c], errors="coerce").values
    ok = ~np.isnan(x)
    if ok.sum() < 100: 
        corr[c] = (np.nan, np.nan); continue
    corr[c] = (np.corrcoef(x[ok], y[ok])[0,1], np.corrcoef(x[ok], ly[ok])[0,1])
c_df = pd.DataFrame(corr, index=["r_y","r_ly"]).T.sort_values("r_y", ascending=False)
print("\nTOP 25 by |corr with y|:")
print(c_df.reindex(c_df.r_y.abs().sort_values(ascending=False).index).head(25).round(3))
print("\nBOTTOM 25 (weakest):")
print(c_df.reindex(c_df.r_y.abs().sort_values().index).head(25).round(3))

# concavity test: linear on raw vs log spend28
def mae(pred, yy): return np.mean(np.abs(pred-yy))
x = pd.to_numeric(df["spend28"], errors="coerce").fillna(0).values
X1 = np.column_stack([np.ones_like(x), x])
b = np.linalg.lstsq(X1, y, rcond=None)[0]; p = X1@b
X2 = np.column_stack([np.ones_like(x), np.log1p(x)])
b2 = np.linalg.lstsq(X2, y, rcond=None)[0]; p2 = X2@b2
X3 = np.column_stack([np.ones_like(x), x, np.log1p(x), np.sqrt(x)])
b3 = np.linalg.lstsq(X3, y, rcond=None)[0]; p3 = X3@b3
print("\nunivariate spend28: MAE lin=%.2f log=%.2f both=%.2f | R2 lin=%.3f log=%.3f" %
      (mae(p,y), mae(p2,y), mae(p3,y), 1-((y-p)**2).sum()/((y-y.mean())**2).sum(), 1-((y-p2)**2).sum()/((y-y.mean())**2).sum()))

# split-half validation of concavity: fit on train snapshot days <=431 (all are train here)
# check per-snapshot: is relationship concave? bin spend28 and show mean y
df["b"] = pd.qcut(df["spend28"].fillna(0), 12, duplicates="drop")
print("\nmean y by spend28 decile:")
print(df.groupby("b", observed=True)["future_spend_4w"].agg(["mean","count"]).round(1))
