
import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
h = agent_api.load_saved("hazard_v1.parquet")
key = ["household_key","snapshot_day"]
hz_cols = [c for c in h.columns if c not in key]
m = e.merge(h, on=key, how="inner")
tt = agent_api.train_targets()
mm = m.merge(tt, on=key, how="left")

for c, v in [("x_spd_sc", mm["spend_28d"]*mm["sc_ratio_mean"].fillna(1.0)),
             ("x_ew_sc", mm["ew_spend_hl28"]*mm["sc_ratio_mean"].fillna(1.0)),
             ("x_spd_carry", mm["spend_28d"]*mm["sc_carry"].fillna(1.0)),
             ("x_usual_sc", mm["usual_4w"]*mm["sc_ratio_mean"].fillna(1.0))]:
    mm[c] = v

feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
tr_fit = day <= 403; tr_val = day == 431
mu, sd = X[tr_fit].mean(0), X[tr_fit].std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, alpha=100.0):
    idx = [feats.index(c) for c in cols]
    A, yy = Xs[tr_fit][:, idx], y[tr_fit]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[tr_val][:, idx] @ w, 0, None)
    return np.abs(p - y[tr_val]).mean()

e011_cols = [c for c in e.columns if c not in key]
print("base:", round(ridge_eval(e011_cols),3))
print("+hz :", round(ridge_eval(e011_cols+hz_cols),3))
print("+hz+inter:", round(ridge_eval(e011_cols+hz_cols+["x_spd_sc","x_ew_sc","x_spd_carry","x_usual_sc"]),3))
print("+inter:", round(ridge_eval(e011_cols+["x_spd_sc","x_ew_sc","x_spd_sc","x_usual_sc"]),3))
