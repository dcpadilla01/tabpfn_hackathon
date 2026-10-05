
import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
h = agent_api.load_saved("hazard_v1.parquet")
key = ["household_key","snapshot_day"]
hz_cols = [c for c in h.columns if c not in key]
m = e.merge(h, on=key, how="inner")
tt = agent_api.train_targets()
mm = m.merge(tt, on=key, how="left")
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, fit_day_max, val_day, alpha=100.0):
    idx = [feats.index(c) for c in cols]
    trf = day <= fit_day_max; trv = day == val_day
    A, yy = Xs[trf][:, idx], y[trf]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[trv][:, idx] @ w, 0, None)
    return np.abs(p - y[trv]).mean()

e011_cols = [c for c in e.columns if c not in key]
pairs = [(375,403),(403,431),(347,375)]
for a in [30,100]:
    d0, d1 = [], []
    for f,v in pairs:
        d0.append(ridge_eval(e011_cols, f, v, a))
        d1.append(ridge_eval(e011_cols+hz_cols, f, v, a))
    print(f"alpha={a}  E011: {[round(x,2) for x in d0]} avg={np.mean(d0):.3f}")
    print(f"alpha={a}  +hz : {[round(x,2) for x in d1]} avg={np.mean(d1):.3f}  delta={np.mean(d1)-np.mean(d0):+.3f}")
