
import agent_api, pandas as pd, numpy as np

names = ['pred_e001','pred_e002','pred_e003_l1','pred_e004','pred_e005','pred_e007',
         'pred_e011','pred_e012','pred_e013','pred_e014','pred_e015','pred_e017']
P = {n: agent_api.load_saved(n + '.parquet') for n in names}
ref = P['pred_e013'][['household_key','snapshot_day']].copy()
M = pd.DataFrame(index=ref.index)
for n in names:
    d = P[n].copy()
    M[n] = ref.merge(d, on=['household_key','snapshot_day'], how='left').prediction.values

tt = agent_api.train_targets()
tr = tt.merge(ref, on=['household_key','snapshot_day'], how='inner')
print('train rows matched:', len(tr), 'days:', sorted(tr.snapshot_day.unique()))
Mtr = ref.merge(tt, on=['household_key','snapshot_day']).merge(
    pd.DataFrame({n: M[n] for n in names}).assign(household_key=ref.household_key, snapshot_day=ref.snapshot_day),
    on=['household_key','snapshot_day'])
y = Mtr.future_spend_4w.values
print('n=', len(y))
for n in names:
    p = Mtr[n].values
    per = []
    for d in [95,123,151,179,207,235,263,291,319,347,375,403,431]:
        m = Mtr.snapshot_day.values==d
        per.append(f"{d%100}:{np.mean(np.abs(p[m]-y[m])):.0f}")
    print(n, 'MAE', round(np.mean(np.abs(p-y)),2), ' '.join(per))
