import agent_api as A, pandas as pd, numpy as np
from scipy.optimize import minimize, nnls
tt = A.train_targets()
o1 = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
o2 = A.load_saved("oof_harness.parquet").merge(tt, on=["household_key","snapshot_day"])
o = o1.merge(o2[["household_key","snapshot_day","oof_med","oof_sq","oof_med_w112"]],
              on=["household_key","snapshot_day"], suffixes=("_x","_y"))
o = o.rename(columns={"oof_med_x":"m1","oof_sq_x":"s1","oof_log":"l1","oof_med_y":"m2","oof_sq_y":"s2","oof_med_w112":"w2"})
y = o.future_spend_4w.values
M = o[["m1","s1","l1","m2","s2","w2"]].values
def mae(w): return np.abs(M@w - y).mean()
res = minimize(mae, np.full(6,1/6), method="Nelder-Mead", options={"maxiter":6000,"fatol":1e-4})
w = np.clip(res.x, 0, None); w = w/w.sum()
print("NM weights:", dict(zip(["m1","s1","l1","m2","s2","w2"], w.round(3))), "MAE:", round(mae(w),3))
w2,_ = nnls(M, y)
print("NNLS weights:", w2.round(3), "sum", w2.sum().round(3), "MAE:", round(mae(w2),3))
print("single m1 MAE:", round(np.abs(o.m1-y).mean(),3))
# per-decile shift on the stack? later. Also check shift stability: fit weights on day<=375, eval on 403/431
tr = o.snapshot_day<=375
res2 = minimize(lambda w: np.abs(M[tr]@w - y[tr]).mean(), np.full(6,1/6), method="Nelder-Mead", options={"maxiter":6000})
w_early = np.clip(res2.x,0,None); w_early/=w_early.sum()
print("early-fit weights:", w_early.round(3), "-> late MAE:", round(mae(w_early),3))
