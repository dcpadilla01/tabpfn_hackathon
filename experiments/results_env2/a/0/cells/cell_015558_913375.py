
import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}
ref = P['pred_e013'][['household_key','snapshot_day']].copy()
ref['k'] = ref.household_key.astype(str) + '_' + ref.snapshot_day.astype(str)
M = pd.DataFrame(index=ref.index)
for n in names:
    d = P[n].copy(); d['k'] = d.household_key.astype(str) + '_' + d.snapshot_day.astype(str)
    M[n] = ref.merge(d[['k','prediction']], on='k', how='left').prediction.values

tt = agent_api.train_targets()
tt['k'] = tt.household_key.astype(str) + '_' + tt.snapshot_day.astype(str)
ref2 = ref[ref.k.isin(set(tt.k))].copy()
Mtr = M.loc[ref2.index]
y = ref2.k.map(dict(zip(tt.k, tt.future_spend_4w))).values
print('train rows:', len(y), 'snapdays:', sorted(ref2.snapshot_day.unique()))

for n in names:
    p = Mtr[n].values
    mae_all = np.mean(np.abs(p-y))
    per = []
    for d in sorted(ref2.snapshot_day.unique()):
        m = ref2.snapshot_day.values==d
        per.append(f"{d}:{np.mean(np.abs(p[m]-y[m])):.0f}")
    print(n, 'MAE', round(mae_all,2), ' '.join(per))
