
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
print("train", Xtr.shape, "val", Xva.shape)

def fit_q(yt, seed, depth=6):
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5,
        learning_rate=0.03, n_estimators=1200, max_depth=depth, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.6, tree_method="hist", n_jobs=8, random_state=seed)
    m.fit(Xtr, yt, sample_weight=w)
    return m.predict(Xva)

t0=time.time()
base = np.maximum(Xva[:, feats.index('spend_28')], 0)
print("baseline spend_28 MAE:", np.abs(base-yva).mean().round(3), f"({time.time()-t0:.0f}s)")

res = {}
t0=time.time(); pA = np.mean([fit_q(ytr, s) for s in (1,2,3)], axis=0); res['A_lin_d6']=pA
print("A lin d6 MAE:", np.abs(pA-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
t0=time.time()
pB = np.mean([np.expm1(np.maximum(fit_q(np.log1p(ytr), s),0)) for s in (1,2,3)], axis=0); res['B_log_d6']=pB
print("B log d6 MAE:", np.abs(pB-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
t0=time.time()
pC = np.mean([np.maximum(fit_q(np.sqrt(ytr), s),0)**2 for s in (1,2,3)], axis=0); res['C_sqrt_d6']=pC
print("C sqrt d6 MAE:", np.abs(pC-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
t0=time.time()
pD = np.mean([fit_q(ytr, s, depth=4) for s in (1,2,3)], axis=0); res['D_lin_d4']=pD
print("D lin d4 MAE:", np.abs(pD-yva).mean().round(3), f"({time.time()-t0:.0f}s)")
for c1 in ['A_lin_d6','B_log_d6','C_sqrt_d6','D_lin_d4']:
    for c2 in ['A_lin_d6','B_log_d6','C_sqrt_d6','D_lin_d4']:
        if c1<c2:
            p=.5*res[c1]+.5*res[c2]
            print("blend",c1,c2, np.abs(p-yva).mean().round(3))
