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
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
def run(cut, tst):
    tri = np.where(df.snapshot_day <= cut)[0]; tei = np.where(df.snapshot_day == tst)[0]
    Xa,ya,Xb,yb,eba,ebb = X[tri],y[tri],X[tei],y[tei],eb[tri],eb[tei]
    def mae(p): return round(np.abs(p-yb).mean(),3)
    pres = eba and None
    pr = np.mean([xq(Xa,ya-eba,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    pl = np.mean([xq(Xa,ya,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    plog = np.expm1(np.mean([xq(Xa,np.log1p(ya)-np.log1p(eba),Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0))
    print(f'[{tst}] plain:{mae(pl)} res:{mae(ebb+pr)} avg(res,plain):{mae((pl+ebb+pr)/2)} 3way+logres:{mae((pl+ebb+pr+plog)/4)}')
    return mae(pl), mae(ebb+pr), mae((pl+ebb+pr)/2), mae((pl+ebb+pr+plog)/4)
r431 = run(403, 431)
r403 = run(375, 403)
r375 = run(347, 375)
print('mean across 3 splits:', np.round(np.mean([r431,r403,r375],axis=0),3))
