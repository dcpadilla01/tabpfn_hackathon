import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
eb = df.exp4w_blend.values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400,lr=0.08):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=lr, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
ebtr, ebte = eb[tr], eb[te]
def mae(p): return round(np.abs(p-yte).mean(),3)
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
def bag(Xa,ya,Xb,resid=False,**kw):
    yy = ya-eb[tr] if False else ya
    return np.mean([xq(Xa,yy,Xb,seed=s,depth=d,mcw=m,**kw) for s,d,m in cfgs], axis=0)
# residual diverse bag
pr = np.mean([xq(Xtr,ytr-ebtr,Xte,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
pres = ebte+pr; print('residual diverse bag:', mae(pres))
# clipping
hi = np.quantile(ytr, 0.995)
print('clip at', hi, '->', mae(np.clip(pres,0,hi)))
# mcw 100/150 diverse
pb2 = np.mean([xq(Xtr,ytr-ebtr,Xte,seed=s,depth=d,mcw=m2) for s,d,m2 in [(7,5,100),(2,5,150),(3,4,100),(4,6,120)]], axis=0)
print('residual mcw100+:', mae(ebte+pb2))
# n=600 lr .05
pb3 = np.mean([xq(Xtr,ytr-ebtr,Xte,seed=s,depth=d,mcw=m,n=600,lr=0.05) for s,d,m in cfgs], axis=0)
print('residual n600 lr.05:', mae(ebte+pb3))
# combine residual bag + plain bag
pdiv = np.mean([xq(Xtr,ytr,Xte,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
print('avg(res,plain):', mae((pres+pdiv)/2))
# check extreme predictions
print('pred max', pres.max(), 'y max', yte.max(), 'n>1000:', (pres>1000).sum())
