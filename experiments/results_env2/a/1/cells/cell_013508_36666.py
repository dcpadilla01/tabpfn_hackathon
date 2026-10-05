
import pandas as pd, numpy as np, xgboost as xgb, time
print('xgb', xgb.__version__)
r = agent_api.load_saved('repro_e5.parquet'); p = agent_api.load_saved('e005_preds.parquet')
m = r.merge(p, on=['household_key','snapshot_day'])
for a in [0.6,0.65,0.7,0.75,0.8]:
    pred = a*m.p13 + (1-a)*m.p11
    print('a=%.2f maxresid %.4f meanresid %.4f' % (a, np.abs(pred-m.prediction).max(), (pred-m.prediction).mean()))
# maybe blend includes clipping
pred = 0.7*m.p13+0.3*m.p11
print('clip200?', np.abs(np.minimum(pred,200)-m.prediction).mean())
print(m.prediction.describe())

F = agent_api.load_saved('allF.parquet')
print('allF dtypes non-numeric:', [c for c in F.columns if F[c].dtype==object])
print('NaN frac overall:', F.isna().mean().mean())
tt = agent_api.train_targets()
tr = F[F.snapshot_day.isin(tt.snapshot_day.unique())]
print('train rows in allF:', len(tr), 'target match?', np.allclose(tr.set_index(['household_key','snapshot_day']).sort_index()['future_spend_4w'].values,
      tt.set_index(['household_key','snapshot_day']).sort_index()['future_spend_4w'].values))
val = F[~F.snapshot_day.isin(tt.snapshot_day.unique())]
print('val rows:', len(val), 'val target nan?', val.future_spend_4w.isna().all())
print('val snapshot days:', sorted(val.snapshot_day.unique()))
