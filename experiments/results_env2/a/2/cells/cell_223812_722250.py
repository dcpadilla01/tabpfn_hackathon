
import numpy as np, pandas as pd
f3 = agent_api.load_saved('feats_v3.parquet')
print('f3 shape', f3.shape)
print('cols:', sorted(f3.columns.tolist()))
print(f3.groupby('snapshot_day').size().to_dict())
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='inner')
print('merged train rows', m.shape)
y = m[agent_api.TARGET]
print(y.describe())
feat_cols = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
absc = m[feat_cols].corrwith(y).abs().sort_values(ascending=False)
print('top |corr| with target:'); print(absc.head(25))
sp28 = [c for c in feat_cols if 'spend' in c.lower() and '28' in c]
print('spend28 candidates:', sp28)
if sp28:
    p = m[sp28[0]].fillna(0).values
    print('naive trailing28 MAE (train):', np.abs(p-y).mean())
print('mean-only MAE (train):', np.abs(np.full(len(y), y.mean())-y).mean())
gm = y.mean(); best=(None,1e9)
for a in np.linspace(0,1.5,31):
    v = np.abs((a*p+(1-a)*gm)-y).mean()
    if v<best[1]: best=(round(a,2),round(v,3))
print('in-sample blend a*trailing28+(1-a)*mean:', best)
