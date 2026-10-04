import numpy as np, pandas as pd
e10 = agent_api.load_saved('e010_rhythm.parquet')
print('e010 cols:', e10.columns.tolist())
tt = agent_api.train_targets()
m = e10.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
# per-snapshot drift of target
g = m.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda s:(s==0).mean()])
g.columns=['mean','median','zero_rate']
print(g)
# correlations with target
num = [c for c in e10.columns if c not in ('household_key','snapshot_day')]
cors = []
for c in num:
    s = m[c]
    if s.dtype==object or str(s.dtype)=='category':
        continue
    cc = s.corr(m.future_spend_4w)
    cors.append((c, cc))
cors = [(c,cc) for c,cc in cors if not np.isnan(cc)]
cors.sort(key=lambda t: abs(t[1]), reverse=True)
print('top |corr| with y:')
for c,cc in cors[:25]: print(f'  {c:24s} {cc: .3f}')
print('bottom:')
for c,cc in cors[-8:]: print(f'  {c:24s} {cc: .3f}')
