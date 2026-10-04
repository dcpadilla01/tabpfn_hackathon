import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e013_denoise.parquet')
print('shape', df.shape)
feat = [c for c in df.columns if c not in ('household_key','snapshot_day')]
print('n_feat', len(feat))
for i in range(0, len(feat), 8):
    print(' | '.join(feat[i:i+8]))
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
y = m['future_spend_4w']
print('y stats', y.describe())
print('zero share', round((y==0).mean(),3))
print(m.groupby('snapshot_day')['future_spend_4w'].agg(mean='mean', med='median', zero=lambda s:(s==0).mean()))
num = m[feat].select_dtypes(include=[np.number])
corr = num.corrwith(y)
print('TOP |corr|:')
print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(30).round(3))
sk = num.skew()
print('MOST SKEWED:'); print(sk.sort_values(ascending=False).head(25).round(1))
print('n |skew|>3:', int((sk.abs()>3).sum()))