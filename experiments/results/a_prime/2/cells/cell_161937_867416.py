import agent_api, pandas as pd, numpy as np
def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags k'.replace(' k','')).copy() if False else agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]
y = tr.future_spend_4w.values
# Try a few ridge-like blends via grid search on train (proxy for val)
best=None
for w1 in [0.3,0.4,0.5,0.6,0.7]:
    for w2 in [0.1,0.2,0.3,0.4]:
        w3 = 1-w1-w2
        if w3<0: continue
        p = w1*tr.spend_28.values+w2*tr.tlag_mean.values+w3*tr.ewma_4.values
        v = mae(y,p)
        if best is None or v<best[0]: best=(v,w1,w2,w3)
print('best blend:', best)
# tlag_2 alone vs spend_28
print('tlag_2 MAE:', round(mae(y, tr.tlag_2.values),2))
# robust: clipped spend_28
for cap in [400,600,800]:
    print(f'clip(spend_28,{cap}) MAE:', round(mae(y, np.minimum(tr.spend_28.values,cap)),2))
# per-day mean target as baseline
day_mean = tr.groupby('snapshot_day')['future_spend_4w'].mean()
p = tr.snapshot_day.map(day_mean).values
print('MAE per-day mean:', round(mae(y,p),2))
# is target autocorrelated with day? check day 431 vs 95
print(tr.groupby('snapshot_day')['future_spend_4w'].mean().diff().round(2).to_dict())
