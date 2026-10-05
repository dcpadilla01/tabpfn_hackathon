import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=0.08, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
def mae(p,yte): return round(np.abs(p-yte).mean(),3)

# diverse bag on internal split (train<=403, test 431)
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
pdiv = np.mean([xq(Xtr,ytr,Xte,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
print('diverse bag:', mae(pdiv,yte))
def calib(p, yte):
    r = minimize(lambda th: np.abs(th[0]*p+th[1]-yte).mean(), [1.0,0.0], method='Nelder-Mead')
    return r.x, r.fun
th, f = calib(pdiv, yte)
print('calib on 431:', th.round(4), 'calibrated MAE:', round(f,3))
# calibration stability: fit on other internal splits, apply to 431 preds
for cut, tst in ((375,403),(347,375),(319,347)):
    tri = np.where(df.snapshot_day <= cut)[0]; tei = np.where(df.snapshot_day == tst)[0]
    pi = np.mean([xq(X[tri],y[tri],X[tei],seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    thi, _ = calib(pi, y[tei])
    print(f'calib fit on {tst}:', thi.round(4), '-> applied to 431 preds MAE:', mae(thi[0]*pdiv+thi[1], yte))
# seed bag + calib for comparison
ps = np.mean([xq(Xtr,ytr,Xte,seed=s) for s in (1,2,3,4,5)], axis=0)
ths, fs = calib(ps, yte)
print('seed bag calib:', ths.round(4), round(fs,3))
