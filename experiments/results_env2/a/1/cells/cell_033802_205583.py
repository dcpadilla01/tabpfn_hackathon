import pandas as pd, numpy as np
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
S = agent_api.load_saved('e016_sqpreds.parquet')
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left').merge(S, on=['household_key','snapshot_day'], how='left')
use_days = [95,123,151,179,207,235,263,291,319,347,375]
oof = tr[tr.snapshot_day.isin(use_days)]
sq = oof[['sq0','sq1','sq2']].mean(axis=1)
print('OOF MAE sq-mean %.2f  |  quantile %.2f' % ((sq-oof.future_spend_4w).abs().mean(), (oof.pred-oof.future_spend_4w).abs().mean()))
print('sq mean %.1f vs quantile mean %.1f vs act %.1f' % (sq.mean(), oof.pred.mean(), oof.future_spend_4w.mean()))
for wq in [0.0,0.25,0.5,0.75,1.0]:
    b = wq*oof.pred + (1-wq)*sq
    print('  wq=%.2f OOF MAE %.2f' % (wq, (b-oof.future_spend_4w).abs().mean()))
# geometric blend
for wq in [0.3,0.5,0.7]:
    g = np.exp(wq*np.log(np.clip(oof.pred,1e-3,None)) + (1-wq)*np.log(np.clip(sq,1e-3,None)))
    print('  geo wq=%.2f OOF MAE %.2f' % (wq, (g-oof.future_spend_4w).abs().mean()))
# held-out late-snap check
h = tr[tr.snapshot_day.isin([403,431])]
sqh = h[['sq0','sq1','sq2']].mean(axis=1)
print('held MAE: quantile %.2f sq %.2f' % ((h.pred-h.future_spend_4w).abs().mean(), (sqh-h.future_spend_4w).abs().mean()))
for wq in [0.0,0.25,0.5,0.75,1.0]:
    b = wq*h.pred + (1-wq)*sqh
    print('  held wq=%.2f MAE %.2f' % (wq, (b-h.future_spend_4w).abs().mean()))