
import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}

ref = P['pred_e013'][['household_key','snapshot_day']].copy()
ref['k'] = ref.household_key.astype(str) + '_' + ref.snapshot_day.astype(str)

M = pd.DataFrame(index=ref.index)
for n in names:
    d = P[n].copy()
    d['k'] = d.household_key.astype(str) + '_' + d.snapshot_day.astype(str)
    m = ref.merge(d[['k','prediction']], on='k', how='left')
    print(n, 'missing:', m.prediction.isna().sum(), 'dup k:', d.k.duplicated().sum())
    M[n] = m.prediction.values

print(M.describe().T[['mean','std','min','max']].round(2))
C = M.corr()
print(C.round(3))
