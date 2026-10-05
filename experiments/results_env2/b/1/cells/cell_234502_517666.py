import agent_api, pandas as pd, numpy as np

e8 = agent_api.load_saved("e008_level_shape.parquet")
tt = agent_api.train_targets()
print("e8 shape", e8.shape, "tt shape", tt.shape)
print(tt['future_spend_4w'].describe())
print("zero share:", (tt['future_spend_4w']==0).mean())
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda s:(s==0).mean()]))

tr = tt.merge(e8, on=["household_key","snapshot_day"], how="left")
print("merged", tr.shape)
y = tr['future_spend_4w'].values

def mae(p): return np.mean(np.abs(p-y))

cands = {
 "sp28": np.expm1(tr['z_log_sp28'].fillna(0).values),
 "sp56h": np.expm1(tr['z_log_sp56'].fillna(0).values)/2,
 "sp84h": np.expm1(tr['z_log_sp84'].fillna(0).values)/3,
 "med4w": tr['z_med4w_hist'].fillna(0).values,
 "meanwk4": tr['z_mean_week_spend_all'].fillna(0).values*4,
}
for k,v in cands.items():
    print(k, "mae=%.2f corr=%.3f" % (mae(v), np.corrcoef(v,y)[0,1]))

# blend optimization on early train snapshots, eval on later
w_days_fit = [95,123,151,179,207,235]
w_days_ev  = [263,291,319,347,375,403,431]
fit = tr['snapshot_day'].isin(w_days_fit).values
ev  = tr['snapshot_day'].isin(w_days_ev).values
keys = list(cands.keys())
X = np.column_stack([cands[k] for k in keys])
from itertools import product
best=None
for w in product(np.linspace(0,1,11), repeat=len(keys)):
    if abs(sum(w)-1)>0.01: continue
    p = X[fit]@np.array(w)
    m = np.mean(np.abs(p-y[fit]))
    if best is None or m<best[0]: best=(m,w)
print("best blend fit:", best)
w=np.array(best[1]); p=X[ev]@w
print("blend ev mae=%.2f (weights %s)" % (np.mean(np.abs(p-y[ev])), dict(zip(keys,w.round(2)))))
print("single best on ev:", {k: round(mae(X[ev][:,i]),2) for i,k in enumerate(keys)})
