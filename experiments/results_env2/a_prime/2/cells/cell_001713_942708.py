import pandas as pd, numpy as np, agent_api

t = agent_api.load_saved('e015_best_pseudo.parquet')
print('shape', t.shape)
print('cols:', list(t.columns))
print(t.dtypes.value_counts())

tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'])
print('merged', m.shape)
y = m['future_spend_4w']
print('zero frac %.3f mean %.1f' % (y.eq(0).mean(), y.mean()))
print('target quantiles:', y.quantile([.5,.75,.9,.95,.99]).round(1).values)

num = m.drop(columns=['future_spend_4w']).select_dtypes('number')
cor = num.corrwith(y).sort_values()
print('\ncorr with target (sorted):')
print(cor.round(3).to_string())
print('\nn features |corr|<0.01:', (cor.abs()<0.01).sum(), ' |corr|<0.02:', (cor.abs()<0.02).sum())
