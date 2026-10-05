import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet'); e5 = A.load_saved('e005_preds.parquet')
ycol='future_spend_4w'
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = F[cols].values.astype(np.float32)
y = F[ycol].values.astype(float); d = F.snapshot_day.values
valm = (d>=459)
t0=time.time()
def qfit(Xtr,ytr,w,Xte,seed,rounds,depth=6):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=depth,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
res={}
for nm, trmask in [('all13',(d<459)&~np.isnan(y)), ('excl403_431',(d<403)&~np.isnan(y))]:
    trm = trmask; w = 0.5**((459 - d[trm])/140.0)
    p = np.zeros(valm.sum())
    for s in (7,17,27):
        p += qfit(X[trm],y[trm],w,X[valm],s,2400)/3
    res[nm]=p
    print(nm,'done',round(time.time()-t0,1),'mean',round(p.mean(),2))
e5v = e5.prediction.values
for nm,p in res.items():
    print(nm,'| corr to e5',round(np.corrcoef(p,e5v)[0,1],4),'| MAEdiff',round(np.abs(p-e5v).mean(),3))
out = F.loc[valm,['household_key','snapshot_day']].copy()
out['p13']=res['all13']; out['p11']=res['excl403_431']
A.save_table(out,'repro_e5.parquet')
print('saved', out.shape)
