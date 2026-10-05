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
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=0.08, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
ebtr, ebte = eb[tr], eb[te]
def mae(p): return round(np.abs(p-yte).mean(),3)
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
def bag(Xa,ya,Xb,**kw): return np.mean([xq(Xa,ya,Xb,seed=s,depth=d,mcw=m,**kw) for s,d,m in cfgs], axis=0)
pdiv = bag(Xtr,ytr,Xte); print('diverse bag base:', mae(pdiv))
# residual learning
pr = bag(Xtr, ytr-ebtr, Xte)
for w in (0.3,0.5,0.7,1.0):
    print(f'residual w={w}:', mae(ebte + w*pr))
# lag4-6 features: need spend in windows s-167..s-140 etc. Compute from transactions via history? Use view at snapshot. 
# Approximate: build lag4..lag6 with build_features is expensive; instead test activity-split models
act = df.spend_84.values
med = np.median(act[tr])
m_hi, m_lo = act[te] > med, act[te] <= med
p_hi = bag(Xtr[act[tr]>med], ytr[act[tr]>med], Xte[m_hi])
p_lo = bag(Xtr[act[tr]<=med], ytr[act[tr]<=med], Xte[m_lo])
psp = np.zeros(len(te)); psp[m_hi] = p_hi; psp[m_lo] = p_lo
print('activity-split:', mae(psp))
# quantile-dispersed mix: 0.5 weight on 0.4/0.6 quantiles (MAE-optimal can differ)
pq = 0.5*xq(Xtr,ytr,Xte,alpha=0.5) + 0.25*xq(Xtr,ytr,Xte,alpha=0.4) + 0.25*xq(Xtr,ytr,Xte,alpha=0.6)
print('q mix .4/.5/.6:', mae(pq))
