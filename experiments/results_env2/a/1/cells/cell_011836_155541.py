import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet')
ycol='future_spend_4w'
allc = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
wf = ['week_of_year','week_sin','week_cos']
variants = {'FULL': allc, 'NOWEEK': [c for c in allc if c not in wf],
            'NOSINCOS': [c for c in allc if c not in ('week_sin','week_cos')],
            'ONLYSINCOS': [c for c in allc if c not in ('week_of_year',)]}
y = F[ycol].values.astype(float); d = F.snapshot_day.values
def qfit(Xtr,ytr,w,Xte,seed,rounds=1200):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=6,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
t0=time.time()
# extrapolation proxy: train <= 347, eval 375/403/431 (unseen raw weeks)
for nm, cols in variants.items():
    X = F[cols].values.astype(np.float32)
    maes=[]
    for ed in (375,403,431):
        trm = (d<=347)&~np.isnan(y); tem=(d==ed)
        w = 0.5**((ed-d[trm])/140.0)
        p = qfit(X[trm],y[trm],w,X[tem],7)
        maes.append(round(np.abs(p-y[tem]).mean(),2))
    print('EXTRAP', nm, maes, round(time.time()-t0,1))
# in-range sanity: train < 403, eval 403; train < 431, eval 431
for nm, cols in variants.items():
    X = F[cols].values.astype(np.float32)
    maes=[]
    for ed in (403,431):
        trm = (d<ed)&~np.isnan(y); tem=(d==ed)
        w = 0.5**((ed-d[trm])/140.0)
        p = qfit(X[trm],y[trm],w,X[tem],7)
        maes.append(round(np.abs(p-y[tem]).mean(),2))
    print('INRANGE', nm, maes, round(time.time()-t0,1))
