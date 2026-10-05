import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, sklearn
print("xgb", xgb.__version__, "| sklearn", sklearn.__version__)
try:
    m = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.4)
    print("tweedie OK")
except Exception as e: print("tweedie ERR", e)
try:
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5)
    print("quantileerror OK")
except Exception as e: print("quantileerror ERR", e)

# 9-learner chunked search on rows where all OOFs exist
oof = A.load_saved("oof_e016_cv.parquet")
oh = A.load_saved("oof_harness.parquet"); op = A.load_saved("oof_pt.parquet")
m = oof.merge(oh, on=["household_key","snapshot_day"], how="left").merge(op, on=["household_key","snapshot_day"], how="left")
L = ["med_v3","med_all","hgbq_v3","hgbq_all","oof_med","oof_sq","oof_med_w112","oof_med_w224","oof_pt"]
mm = m.dropna(subset=L)
print("rows with all 9 OOFs:", len(mm), "snapshot days:", sorted(mm.snapshot_day.unique()))
y = mm["y"].values; P = mm[L].values
for k,l in enumerate(L): print(f"{l:12s} MAE {np.abs(y-P[:,k]).mean():.4f}")
rng = np.random.default_rng(1)
best_mae, best_w = 1e9, None
for it in range(30):
    W = rng.dirichlet(np.ones(9)*0.5, size=4000)
    PW = P @ W.T
    maes = np.abs(y[:,None]-PW).mean(axis=0)
    b = maes.argmin()
    if maes[b] < best_mae: best_mae, best_w = maes[b], W[b].copy()
print("best random 9-learner w:", dict(zip(L, np.round(best_w,3))), "OOF MAE", round(best_mae,4))
# refine
def mae_w(w): return np.abs(y - P @ w).mean()
bw = best_w.copy(); bm = mae_w(bw)
for step in [0.1,0.05,0.02,0.01,0.005]:
    improved=True
    while improved:
        improved=False
        for i in range(9):
            for j in range(9):
                if i==j or bw[j]<step: continue
                w2=bw.copy(); w2[i]+=step; w2[j]-=step
                m2=mae_w(w2)
                if m2<bm: bw,bm=w2,m2; improved=True
print("refined:", dict(zip(L, np.round(bw,3))), "OOF MAE", round(bm,4))
