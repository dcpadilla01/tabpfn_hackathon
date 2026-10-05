import pandas as pd, numpy as np
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
S = agent_api.load_saved('e016_sqpreds.parquet')
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left').merge(S, on=['household_key','snapshot_day'], how='left')
use_days = [95,123,151,179,207,235,263,291,319,347,375]
oof = tr[tr.snapshot_day.isin(use_days)]
sq = oof[['sq0','sq1','sq2']].mean(axis=1)
print('OOF MAE sq-mean %.2f' % ((sq-oof.future_spend_4w).abs().mean()))
print('sq mean %.1f vs act %.1f' % (sq.mean(), oof.future_spend_4w.mean()))
for wq in [0.0,0.25,0.5,0.75,1.0]:
    b = wq*oof.future_spend_4w*0 + wq*0  # placeholder no
for wq in [0.0,0.25,0.5,0.75,1.0]:
    b = (1-wq)*sq  # need quantile preds; recompute via saved e5? quantile preds exist only for val.
print('note: quantile OOF preds not saved for train; use held-out 403/431 instead')
h = tr[tr.snapshot_day.isin([403,431])]
sqh = h[['sq0','sq1','sq2']].mean(axis=1)
print('held MAE: sq %.2f' % (sqh-h.future_spend_4w).abs().mean())
print('held sq mean %.1f act %.1f' % (sqh.mean(), h.future_spend_4w.mean()))
agent_api.save_table(h[['household_key','snapshot_day','sq0','sq1','sq2','future_spend_4w']], 'e016_held.parquet')