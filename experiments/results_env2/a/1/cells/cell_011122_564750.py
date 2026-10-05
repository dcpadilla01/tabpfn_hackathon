import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet')
ycol='future_spend_4w'
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = F[cols].values.astype(np.float32)
y = F[ycol].values.astype(float); d = F.snapshot_day.values
t0=time.time()
def qfit(Xtr,ytr,w,Xte,seed,rounds=1200,depth=6):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=depth,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
def ev(ed, cut, seeds=(7,17)):
    trm = (d<cut)&~np.isnan(y); tem = (d==ed)
    w = 0.5**((ed - d[trm])/140.0)
    p = np.zeros(tem.sum())
    for s in seeds: p += qfit(X[trm],y[trm],w,X[tem],s)/len(seeds)
    ya = y[tem]
    return np.abs(p-ya).mean(), p.mean(), ya.mean(), p
for ed in (403,431):
    for nm,cut in [('all',(ed-28+1)),('excl2',(ed-84+1))]:
        mae,pm,yam,_ = ev(ed,cut)
        print(f'day {ed} cut<={cut-1} [{nm}] MAE {mae:.3f} predmean {pm:.1f} actualmean {yam:.1f} bias {pm-yam:+.1f}')
print('elapsed',round(time.time()-t0,1))
