
import agent_api, pandas as pd, numpy as np
print(agent_api.snapshot_days())
tt = agent_api.train_targets()
t = tt.future_spend_4w
print('train rows:', len(tt))
print('target mean %.2f std %.2f median %.2f zero-frac %.3f' % (t.mean(), t.std(), t.median(), (t==0).mean()))
K = agent_api.KEYS
h2 = agent_api.load_saved('hist_v2.parquet'); m2 = agent_api.load_saved('mkt_v2.parquet')
print('hist_v2', h2.shape, 'mkt_v2', m2.shape)
print('hist_v2 cols:', h2.columns.tolist())
print('mkt_v2 extra cols:', [c for c in m2.columns if c not in h2.columns])
df = tt.merge(h2, on=K, how='left')
extra = [c for c in m2.columns if c not in df.columns and c not in K]
if extra: df = df.merge(m2[extra+K], on=K, how='left')
num = df.select_dtypes(include=[np.number])
cor = num.corr()['future_spend_4w'].drop('future_spend_4w')
cor = cor.reindex(cor.abs().sort_values(ascending=False).index)
print(cor.head(45).round(3).to_string())
