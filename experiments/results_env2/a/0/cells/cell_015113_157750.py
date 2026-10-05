
import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}

# check alignment
base = P['pred_e013'][['household_key','snapshot_day']].copy()
base = base.sort_values(['household_key','snapshot_day']).reset_index(drop=True)
for n in names:
    q = P[n][['household_key','snapshot_day']].sort_values(['household_key','snapshot_day']).reset_index(drop=True)
    assert q.equals(base), n
print('all aligned, rows:', len(base))

M = pd.DataFrame({n: P[n]['prediction'].values for n in names})
print(M.describe().T[['mean','std','min','max']])

C = M.corr()
print(C.round(4))
