
import agent_api, pandas as pd, numpy as np, re
from collections import defaultdict

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
print('E009 shape', t.shape)
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
pref = defaultdict(list)
for c in cols:
    p = re.split(r'[\d_]', c)[0]
    pref[p].append(c)
for p in sorted(pref, key=lambda p: -len(pref[p])):
    print(f'{p:22s} {len(pref[p]):4d}  e.g. {pref[p][:5]}')

m = t.merge(tt, on=['household_key','snapshot_day'])
print('train rows', len(m))
y = m['future_spend_4w'].values
print('pct', np.percentile(y,[0,25,50,75,90,95,99]).round(1), 'mean', round(y.mean(),1), 'zero%', round((y==0).mean(),3))
print('MAE pred0', round(np.abs(y).mean(),2))

num = m[cols].select_dtypes(include=[np.number])
print('numeric feats', num.shape[1], 'nonnumeric', len(cols)-num.shape[1])
print('nonnum dtypes:', set(map(str, m[cols].dtypes)))
corr = num.corrwith(m['future_spend_4w'])
cs = corr.abs().sort_values(ascending=False)
print(cs.head(30).round(3).to_string())
