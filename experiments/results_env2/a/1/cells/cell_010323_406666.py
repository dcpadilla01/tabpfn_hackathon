import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet'); e5 = A.load_saved('e005_preds.parquet')
ycol='future_spend_4w'
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = F[cols].values.astype(np.float32)
y = F[ycol].values.astype(float); d = F.snapshot_day.values
valm = ~np.isnan(y)
t0=time.time()
def qfit(Xtr,ytr,w,Xte,seed,rounds,depth=6):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=depth,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
trm = (d<459)&~np.isnan(y); w = 0.5**((459-d[trm])/140.0)
p13 = np.zeros(valm.sum())
for s in (7,17,27):
    p13 += qfit(X[trm],y[trm],w,X[valm],s,2400)/3
print('repro-all13 done', round(time.time()-t0,1))
trm11 = (d<403)&~np.isnan(y); w11 = 0.5**((459-d[trm11])/140.0)
p11 = np.zeros(valm.sum())
for s in (7,17,27):
    p11 += qfit(X[trm11],y[trm11],w11,X[valm],s,2400)/3
print('repro-11 done', round(time.time()-t0,1))
e5v = e5.prediction.values
for nm,p in [('all13',p13),('excl403/431',p11)]:
    print(nm,'mean',round(p.mean(),2),'| e5 mean',round(e5v.mean(),2),'| corr',round(np.corrcoef(p,e5v)[0,1],4),'| MAEdiff',round(np.abs(p-e5v).mean(),3))
