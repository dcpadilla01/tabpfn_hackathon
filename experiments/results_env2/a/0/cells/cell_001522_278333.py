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
tri = np.where(df.snapshot_day <= 431)[0]
tei = np.where(df.snapshot_day >= 459)[0]
Xa,ya,Xb,eba = X[tri],y[tri],X[tei],eb[tei]
pr = np.mean([xq(Xa,ya-eba,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
pl = np.mean([xq(Xa,ya,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
pred = 0.4*(eba+pr) + 0.6*pl
out = df.iloc[tei][['household_key','snapshot_day']].copy()
out['prediction'] = pred
path = agent_api.save_table(out, 'pred_e013.parquet')
print('saved', path, out.shape, 'pred range', pred.min().round(1), pred.max().round(1))
