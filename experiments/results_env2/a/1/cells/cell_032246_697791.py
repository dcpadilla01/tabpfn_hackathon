import agent_api, numpy as np, pandas as pd, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
allF = agent_api.load_saved("allF.parquet")
feats = [c for c in allF.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
tr_days = [123,151,179,207,235,263,291,319,347,375]
tr = allF[allF.snapshot_day.isin(tr_days)]
va = allF[allF.snapshot_day==403]
Xtr, ytr = tr[feats].values, tr.future_spend_4w.values
Xva, yva = va[feats].values, va.future_spend_4w.values
w = 0.5 ** ((375 - tr.snapshot_day.values)/140.0)

def fit_q(yt, seed, depth=6, lr=0.03, n=1200):
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
        learning_rate=lr, n_estimators=n, max_depth=depth, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.6, tree_method="hist", n_jobs=8, random_state=seed)
    m.fit(Xtr, yt, sample_weight=w)
    return m.predict(Xva)

# E: linear d5
t0=time.time()
pE = np.mean([fit_q(ytr, s, depth=5) for s in (1,2,3)], axis=0)
print("E lin d5 MAE:", np.abs(pE-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
# F: linear d6 longer rounds
t0=time.time()
pF = np.mean([fit_q(ytr, s, depth=6, n=2400) for s in (1,2,3)], axis=0)
print("F lin d6 n2400 MAE:", np.abs(pF-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
# G: linear d6, min_child_weight 30
t0=time.time()
def fit_q2(yt, seed, mcw=30):
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
        learning_rate=0.03, n_estimators=1200, max_depth=6, min_child_weight=mcw,
        subsample=0.8, colsample_bytree=0.6, tree_method="hist", n_jobs=8, random_state=seed)
    m.fit(Xtr, yt, sample_weight=w)
    return m.predict(Xva)
pG = np.mean([fit_q2(ytr, s) for s in (1,2,3)], axis=0)
print("G lin d6 mcw30 MAE:", np.abs(pG-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
# blends with d4
print("blend d4+d5:", np.abs(.5*pD+.5*pE-yva).mean().round(3))
print("blend d4+d6:", np.abs(.5*pD+.5*pA-yva).mean().round(3))
print("blend d5+d6:", np.abs(.5*pE+.5*pA-yva).mean().round(3))
print("blend d4+2d6:", np.abs((pD+2*pA)/3-yva).mean().round(3))