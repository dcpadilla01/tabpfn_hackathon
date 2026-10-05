
import agent_api, pandas as pd, numpy as np
e5 = agent_api.load_saved('e005_decay_gapcv.parquet')
e6 = agent_api.load_saved('e006_momentum_seasonal.parquet')
print('E5 columns (%d):' % len(e5.columns))
print(e5.columns.tolist())
print('E6 extra:', [c for c in e6.columns if c not in e5.columns])
tt = agent_api.train_targets()
df = e6.merge(tt, on=['household_key','snapshot_day'])
print('merged', df.shape)
y = df['future_spend_4w']
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print('zero share %.3f' % (y==0).mean())
num = df.drop(columns=['household_key','snapshot_day','future_spend_4w']).select_dtypes(include=[np.number])
cor = num.corrwith(y)
cor = cor.reindex(cor.abs().sort_values(ascending=False).index)
print(cor.round(3).to_string())
g = df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count'])
print(g.round(1).to_string())
print('global mean pred MAE %.2f' % (y - y.mean()).abs().mean())
for c in cor.index[:12]:
    print('naive MAE by %-16s %.2f' % (c, (y - df[c]).abs().mean()))
