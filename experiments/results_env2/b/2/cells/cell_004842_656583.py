
import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
h = agent_api.load_saved("hazard_v1.parquet")
key = ["household_key","snapshot_day"]
m = e.merge(h, on=key, how="inner")
tt = agent_api.train_targets()
mm = m.merge(tt, on=key, how="left")

feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values

tr_fit = day <= 403   # pseudo-train
tr_val = day == 431   # pseudo-val (last train snapshot)
print("fit rows:", tr_fit.sum(), "val rows:", tr_val.sum())

mu, sd = X[tr_fit].mean(0), X[tr_fit].std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(mask, alpha=100.0, logt=False, w=None):
    A, yy = Xs[tr_fit][:, mask], y[tr_fit]
    if logt: yy = np.log1p(yy)
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = Xs[tr_val][:, mask] @ w
    if logt: p = np.expm1(p)
    p = np.clip(p, 0, None)
    return np.abs(p - y[tr_val]).mean()

hz = np.array([c in [c for c in h.columns if c not in key] for c in feats])
allm = np.ones(len(feats), bool)
for a in [30, 100, 300]:
    print(f"alpha={a}: E011={ridge_eval(allm,a):.2f}  E011+hz={ridge_eval(allm|hz,a):.2f}")
print("logt: E011=", round(ridge_eval(allm,100,logt=True),2), " E011+hz=", round(ridge_eval(allm|hz,100,logt=True),2))
