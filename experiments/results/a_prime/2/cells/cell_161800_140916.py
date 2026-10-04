import agent_api, pandas as pd, numpy as np
def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
tt = agent_api.train_targets()
e = agent_api.load_saved('e009_ewma_longlags.parquet')
m = e.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431].copy()
print('train rows', len(tr))
print('MAE spend_84:', round(mae(tr.future_spend_4w, tr.spend_84),2))
print('MAE spend_28:', round(mae(tr.future_spend_4w, tr.spend_28),2))
print('MAE ewma_4:', round(mae(tr.future_spend_4w, tr.ewma_4),2))
print('MAE mean(train):', round(mae(tr.future_spend_4w, np.full(len(tr), tr.future_spend_4w.mean())),2))
# hybrid: 0 for zero28 else spend_84
pred = np.where(tr.spend_28==0, 0, tr.spend_84)
print('MAE hybrid zero28|spend84:', round(mae(tr.future_spend_4w, pred),2))
# per-snapshot-day mean target (calendar effect)
day_mean = tr.groupby('snapshot_day')['future_spend_4w'].mean()
print(day_mean.round(1).to_dict())
# how much of MAE comes from zero28 rows? if we predict their conditional mean:
zmean = tr[tr.spend_28==0].future_spend_4w.mean()
pred2 = np.where(tr.spend_28==0, zmean, tr.spend_84)
print('MAE hybrid zero28-mean|spend84:', round(mae(tr.future_spend_4w, pred2),2))
# error decomposition proxy: MAE within zero28 rows using best constant
z = tr[tr.spend_28==0]
print('zero28 rows: MAE with constant mean:', round(mae(z.future_spend_4w, np.full(len(z), zmean)),2),
      'MAE with 0:', round(mae(z.future_spend_4w, np.zeros(len(z))),2))
a = tr[tr.spend_28>0]
amean = a.future_spend_4w.mean()
print('active rows: MAE const:', round(mae(a.future_spend_4w, np.full(len(a), amean)),2),
      'MAE spend_84:', round(mae(a.future_spend_4w, a.spend_84),2))
