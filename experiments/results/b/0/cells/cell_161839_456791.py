import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
rfm = A.load_saved("rfm28.parquet").merge(tt, on=["household_key","snapshot_day"])
cols = ["spend28","trips28","recency"]

def prep(df, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    return ((X-mu)/sd).clip(-10,10).fillna(0).values

d = rfm
Xs = np.column_stack([np.ones(len(d)), prep(d, cols)])
y = d["future_spend_4w"].values
tr = d.snapshot_day.isin(tr_days).values; m = ~tr
Xtr, ytr = Xs[tr], y[tr]

def irls(X, y, delta=None, n_iter=30, lam=1.0):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        r = y - X@b
        s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
        dl = delta if delta else 1.345*s
        w = np.minimum(1.0, dl/np.maximum(np.abs(r), 1e-9))
        W = np.diag(w) if len(w)<5 else None
        # weighted solve with ridge
        Xw = X * w[:,None]
        b = np.linalg.solve(X.T@Xw + lam*np.eye(X.shape[1]), Xw.T@y)
    return b

# 1) plain ridge (baseline)
b = np.linalg.solve(Xtr.T@Xtr+1*np.eye(4), Xtr.T@ytr)
print("ridge          :", np.abs(Xs[m]@b - y[m]).mean().round(3), "(harness 69.595)")
# 2) Huber/LAD-ridge
for dl in [None, 30, 60]:
    b = irls(Xtr, ytr, delta=dl)
    print(f"huber delta={dl}:", np.abs(Xs[m]@b - y[m]).mean().round(3))
# 3) recency-weighted ridge (weight ~ snapshot day)
w_snap = d.snapshot_day.values / 431.0
Xw = Xtr * w_snap[tr][:,None]
b = np.linalg.solve(Xtr.T@Xw + 1*np.eye(4), Xw.T@ytr)
print("recency-w ridge:", np.abs(Xs[m]@b - y[m]).mean().round(3))
# 4) both
b = irls(Xtr, ytr, delta=60)
r = ytr - Xtr@b; w = np.minimum(1.0, 60/np.maximum(np.abs(r),1e-9)) * w_snap[tr]
Xw = Xtr * w[:,None]
b = np.linalg.solve(Xtr.T@Xw + 1*np.eye(4), Xw.T@ytr)
print("huber+recency  :", np.abs(Xs[m]@b - y[m]).mean().round(3))
# 5) sample weights by day as pure weighting of squared loss (no robustness)
for pw in [0.5, 1.0, 2.0]:
    w = (d.snapshot_day.values/431.0)**pw
    Xw = Xtr * w[tr][:,None]
    b = np.linalg.solve(Xtr.T@Xw + 1*np.eye(4), Xw.T@ytr)
    print(f"day^pw={pw} ridge:", np.abs(Xs[m]@b - y[m]).mean().round(3))
