
import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
h = agent_api.load_saved("hazard_v1.parquet")
key = ["household_key","snapshot_day"]
print("dup keys hazard:", h[key].duplicated().sum(), "| hazard dtypes ok:", h.drop(columns=key).dtypes.unique())

m = e.merge(h, on=key, how="inner")
print("merged:", m.shape, "== e011 rows:", len(e))
print("hazard cols:", [c for c in h.columns if c not in key])

tt = agent_api.train_targets()
mm = m.merge(tt, on=key, how="left")
tr = mm["future_spend_4w"].notna().values
print("train rows:", tr.sum(), "val rows:", (~tr).sum())

# quick proxy: ridge on standardized features, fit train -> MAE val
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = mm[feats].astype(float).values
X = np.nan_to_num(X, nan=0.0)
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
Xs = (X - mu) / sd
y = mm["future_spend_4w"].values

def ridge_eval(cols_mask, alpha=100.0, logt=False):
    A = Xs[tr][:, cols_mask]; yy = y[tr]
    if logt: yy = np.log1p(yy)
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = Xs[~tr][:, cols_mask] @ w
    if logt: p = np.expm1(p)
    p = np.clip(p, 0, None)
    return np.abs(p - y[~tr]).mean()

i28 = feats.index("spend_28d")
print("proxy ridge MAE  spend_28d only:", round(ridge_eval(np.eye(len(feats))[i28].astype(bool)),2))
allm = np.ones(len(feats), bool)
hz = np.array([c in [c for c in h.columns if c not in key] for c in feats])
print("proxy ridge MAE  E011(all):", round(ridge_eval(allm),2))
print("proxy ridge MAE  E011+hazard:", round(ridge_eval(allm | hz),2))
print("proxy ridge MAE  E011+hazard (logt):", round(ridge_eval(allm | hz, logt=True),2))
print("proxy ridge MAE  E011 (logt):", round(ridge_eval(allm, logt=True),2))
