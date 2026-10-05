import agent_api as A, pandas as pd, numpy as np
from scipy.optimize import minimize
tt = A.train_targets()
o1 = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
o2 = A.load_saved("oof_harness.parquet").merge(tt, on=["household_key","snapshot_day"])
o = o1.merge(o2[["household_key","snapshot_day","oof_med_y","oof_sq_y","oof_med_w112"]],
              on=["household_key","snapshot_day"], suffixes=("_x","_y"))
o = o.rename(columns={"oof_med_x":"m1","oof_sq_x":"s1","oof_log":"l1","oof_med_y":"m2","oof_sq_y":"s2","oof_med_w112":"w2"})
y = o.future_spend_4w.values
M = o[["m1","s1","l1","m2","s2","w2"]].values
def mae(w, M, y):
    p = M @ w
    return np.abs(p-y).mean()
w0 = np.zeros(M.shape[1]); w0[0]=1
res = minimize(mae, np.full(6,1/6), args=(M,y), method="Nelder-Mead",
               options={"maxiter":4000,"xatol":1e-3,"fatol":1e-3})
w = res.x
print("weights:", dict(zip(["m1","s1","l1","m2","s2","w2"], w.round(3))))
print("stacked OOF MAE:", round(mae(w,M,y),3), " (single med1:", round(np.abs(o.m1-y).mean(),3), ")")
# simpler: nonneg least squares on MAE via repeated Nelder-Mead with clipping
from scipy.optimize import nnls
w2,_ = nnls(M, y)
print("NNLS weights:", w2.round(3), "MAE:", round(np.abs(M@w2-y).mean(),3))
