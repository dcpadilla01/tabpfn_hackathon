import agent_api as A, pandas as pd, numpy as np
pd.set_option("display.width", 220)
f3 = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
oo = A.load_saved("oof_e008.parquet")
m = tt.merge(oo, on=["household_key","snapshot_day"]).merge(f3[["household_key","snapshot_day","spend_28","spend_84","active_28","days_since_last","baskets_28","gap_mean_112","gap_std_112","active_84"]], on=["household_key","snapshot_day"])
y = m["future_spend_4w"].values
def mae(p): return round(np.mean(np.abs(np.asarray(p)-y)),3)
# Is sq better than med on high spend?
for lo,hi in [(0,50),(50,150),(150,300),(300,600),(600,10**9)]:
    s = m[(m.spend_28>=lo)&(m.spend_28<hi)]
    print(f"s28 [{lo},{hi}) n={len(s)}", 
          "sq", round(np.abs(s.oof_sq-s.future_spend_4w).mean(),1),
          "med", round(np.abs(s.oof_med-s.future_spend_4w).mean(),1),
          "log", round(np.abs(s.oof_log-s.future_spend_4w).mean(),1),
          "y_mean", round(s.future_spend_4w.mean(),1))
# overall best weighted blend using scipy
from scipy.optimize import minimize
def f(w):
    p = w[0]*m.oof_sq + w[1]*m.oof_med + w[2]*m.oof_log
    return np.mean(np.abs(p-y))
r = minimize(f, [0.33,0.34,0.33], method="Nelder-Mead")
print("best simplex blend:", np.round(r.x,3), round(r.fun,3))
# blend of sq and log only
def f2(w): return np.mean(np.abs((w*m.oof_sq+(1-w)*m.oof_log)-y))
r2 = minimize(f2, [0.5], method="Nelder-Mead")
print("sq+log blend:", round(r2.x[0],3), round(r2.fun,3))
