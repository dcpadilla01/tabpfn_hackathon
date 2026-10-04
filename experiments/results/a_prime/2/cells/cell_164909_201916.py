import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
X = df[feat_cols].copy()
for c in X.columns:
    if str(X[c].dtype)=='category' or X[c].dtype==bool:
        X[c]=X[c].astype('category').cat.codes.replace(-1,np.nan)
Xv = X.astype(np.float64).values
print('nan count', np.isnan(Xv).sum())
inf_mask = np.isinf(Xv)
print('inf count', inf_mask.sum())
cols_inf = [feat_cols[j] for j in range(len(feat_cols)) if inf_mask[:,j].any()]
print('cols with inf:', cols_inf)
y = df.future_spend_4w.values.astype(float)
print('y nan', np.isnan(y).sum())
# check per column nan fraction >0.9
naf = np.isnan(Xv).mean(0)
print('cols >90% nan:', [feat_cols[j] for j in range(len(feat_cols)) if naf[j]>0.9])
