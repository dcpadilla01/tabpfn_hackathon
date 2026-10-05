
import agent_api, pandas as pd, numpy as np

names = ['pred_e004','pred_e005','pred_e007','pred_e008','pred_e011',
         'pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}
ref = P['pred_e013'][['household_key','snapshot_day']].copy()
M = pd.DataFrame(index=ref.index)
for n in names:
    d = P[n]
    M[n] = ref.merge(d, on=['household_key','snapshot_day'], how='left').prediction.values
print('NaNs per model:', M.isna().sum().values, 'rows:', len(M))

blend = M.mean(axis=1).clip(lower=0).values
print('blend mean/std/min/max:', blend.mean().round(2), blend.std().round(2), blend.min().round(2), blend.max().round(2))

out = ref.copy()
out['prediction'] = blend
p = agent_api.save_table(out, 'pred_e018.parquet')
print('saved:', p)
