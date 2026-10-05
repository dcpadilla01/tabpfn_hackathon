import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
t0=time.time()
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
M = F.merge(W[['household_key','snapshot_day','sp28_yag','sp84_yag']], on=['household_key','snapshot_day'], how='left')
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
# year-ago pace ratios (NaN when no year-ago history; XGBoost handles NaN)
M['r28yag'] = np.where(M.sp28_yag>0, M.spend_28/M.sp28_yag, np.nan)
M['r84yag'] = np.where(M.sp84_yag>0, M.spend_84/M.sp84_yag, np.nan)
M['r28yag_c'] = M.r28yag.clip(0,5); M['r84yag_c'] = M.r84yag.clip(0,5)
colsR = cols + ['r28yag','r84yag','r28yag_c','r84yag_c']
y = M[ycol].values.astype(float); d = M.snapshot_day.values
tr_days_all = [95,123,151,179,207,235,263,291,319,347,375,403,431]
def qfit(Xtr,ytr,w,Xte,seed,rounds):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=6,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
# local CV at 403/431 (1 seed, 1200 rounds)
for nm, cl in [('BASE',cols), ('RATIO',colsR)]:
    X = M[cl].values.astype(np.float32); maes=[]
    for ed in (403,431):
        trm=(d<ed)&~np.isnan(y); tem=(d==ed); w=0.5**((ed-d[trm])/140.0)
        p=qfit(X[trm],y[trm],w,X[tem],7,1200)
        maes.append(round(np.abs(p-y[tem]).mean(),3))
    print(nm, maes, round(time.time()-t0,1))
# final: E005 recipe (all13, decay140, 2400 rounds, 3 seeds) on RATIO features -> val preds
X = M[colsR].values.astype(np.float32)
valm = (d>=459); trm=(d<459)&~np.isnan(y)
w = 0.5**((459-d[trm])/140.0)
pv = np.zeros(valm.sum())
for s in (7,17,27): pv += qfit(X[trm],y[trm],w,X[valm],s,2400)/3
out = M.loc[valm,['household_key','snapshot_day']].copy(); out['prediction']=pv
A.save_table(out,'e010_preds.parquet')
print('saved e010', out.shape, 'predmean', round(pv.mean(),2), 'elapsed', round(time.time()-t0,1))
