
import pandas as pd, numpy as np

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for c in cat_cols:
    d = pd.get_dummies(X[c].astype("category"), prefix=c[:10], dummy_na=True)
    Xn = pd.concat([Xn, d.astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64)
n = len(y)

# correlations (fixed)
cors = {}
for c in Xn.columns:
    v = Xn[c].values
    if np.nanstd(v) > 1e-12:
        cors[c] = abs(float(np.corrcoef(v, y)[0,1]))
top = sorted(cors.items(), key=lambda kv: -kv[1])[:25]
print("Top-25 |corr| with target:")
for c, val in top: print(f"  {c:22s} {val:.3f}")

# --- structural tests ---
idx = np.random.RandomState(0).permutation(n); K=5
def cv_mae(pred_fn):
    s = 0.0
    for k in range(K):
        va = idx[k::K]; tr = np.setdiff1d(idx, va)
        p = pred_fn(tr, va)
        s += np.abs(y[va]-p).sum()
    return round(s/n, 2)

def ridge_fit(Xtr, ytr, a=100):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    A=np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
    return np.linalg.solve(Gg, A.T@ytr), mu, sd

# 1) plain ridge
def f_ridge(tr, va):
    w, mu, sd = ridge_fit(Xv[tr], y[tr])
    return np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
print("\nridge all:", cv_mae(f_ridge))

# 2) two-piece: inactive (sp28==0) -> constant, active -> ridge
sp28 = df["sp28"].values.astype(float)
inact = sp28 <= 0
def f_2piece(tr, va):
    c = y[tr][inact[tr]].mean() if inact[tr].any() else 0.0
    w, mu, sd = ridge_fit(Xv[tr][~inact[tr]], y[tr][~inact[tr]])
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    p[inact[va]] = c
    return p
print("two-piece (const for sp28==0):", cv_mae(f_2piece))

# 3) ridge on log target, exp back (two-piece too)
def f_2piece_log(tr, va):
    c = np.expm1(np.log1p(y[tr][inact[tr]]).mean()) if inact[tr].any() else 0.0
    w, mu, sd = ridge_fit(Xv[tr][~inact[tr]], np.log1p(y[tr][~inact[tr]]))
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    p = np.clip(np.expm1(p), 0, None); p[inact[va]] = c
    return p
print("two-piece log-target:", cv_mae(f_2piece_log))

# 4) blend ridge with med4w hist
med4 = df["z_med4w_hist"].values.astype(float)
def f_blend(tr, va):
    w, mu, sd = ridge_fit(Xv[tr], y[tr])
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    return 0.5*p + 0.5*med4[va]
print("blend ridge+med4w 50/50:", cv_mae(f_blend))

# 5) quantile cap
def f_cap(tr, va):
    w, mu, sd = ridge_fit(Xv[tr], y[tr])
    p = np.hstack([(Xv[va]-mu)/sd, np.ones((len(va),1))]) @ w
    hi = np.quantile(y[tr], 0.97)
    return np.clip(p, 0, hi)
print("ridge capped at q97:", cv_mae(f_cap))
