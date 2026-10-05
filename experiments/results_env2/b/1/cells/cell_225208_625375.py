
import pandas as pd, numpy as np

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
print("merged", df.shape)
y = df[TARGET].values.astype(float)

feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for c in cat_cols:
    d = pd.get_dummies(X[c].astype("category"), prefix=c[:10], dummy_na=True)
    Xn = pd.concat([Xn, d.astype(float)], axis=1)
med = Xn.median()
Xn = Xn.fillna(med)
print("design", Xn.shape, "cats:", cat_cols)

print("\nBaselines on train rows:")
print("  MAE predict-0   :", np.abs(y).mean().round(2))
print("  MAE global mean :", np.abs(y-y.mean()).mean().round(2))
if "sp28" in df: print("  MAE sp28 (last4w):", np.abs(y-df["sp28"]).mean().round(2))
if "sp84_rate" in df: print("  MAE sp84_rate*4 :", np.abs(y-df["sp84_rate"]*4).mean().round(2))
if "z_med4w_hist" in df: print("  MAE med4w hist  :", np.abs(y-df["z_med4w_hist"]).mean().round(2))

# ridge CV
idx = np.random.RandomState(0).permutation(len(y)); K=5
def ridge_cv(Xv, yv, alphas=(1,10,100,1000), logt=False):
    n=len(yv); maes={a:0.0 for a in alphas}
    yt = np.log1p(yv) if logt else yv
    for k in range(K):
        va = idx[k::K]; tr = np.setdiff1d(idx, va)
        mu=Xv[tr].mean(0); sd=Xv[tr].std(0)+1e-9
        A=np.hstack([(Xv[tr]-mu)/sd, np.ones((len(tr),1))])
        B=np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))])
        G=A.T@A; p=G.shape[0]
        for a in alphas:
            Gg=G.copy(); Gg[np.arange(p-1),np.arange(p-1)]+=a
            w=np.linalg.solve(Gg, A.T@yt[tr])
            pred=B@w
            if logt: pred=np.clip(np.expm1(pred),0,None)
            maes[a]+=np.abs(yv[va]-pred).sum()
    return {a: round(m/n,2) for a,m in maes.items()}

Xv = Xn.values.astype(np.float64)
print("\nRidge CV MAE (raw target):", ridge_cv(Xv, y))
print("Ridge CV MAE (log target):", ridge_cv(Xv, y, logt=True))

# feature correlations with target
cors = {}
for c in Xn.columns:
    cors[c] = abs(np.corrcoef(Xn[c].values, y)[0,1]) if Xn[c].std()>0 else 0
top = sorted(cors.items(), key=lambda kv:-kv[1])[:25]
print("\nTop-25 |corr| with target:")
for c,v in top: print(f"  {c:24s} {v:.3f}")
