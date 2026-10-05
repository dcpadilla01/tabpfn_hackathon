
import agent_api as A, numpy as np, pandas as pd

oof = A.load_saved('oof_e013.parquet')
print('oof shape', oof.shape, 'cols', oof.columns.tolist())
print(oof.head(3))
print(oof.groupby('snapshot_day').size())

tt = A.train_targets()
print('targets', tt.shape)
m = tt.merge(oof, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
pcol = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')][0]
print('pred col', pcol)
print('OOF MAE e013:', np.abs(m[pcol]-m.future_spend_4w).mean())
g = m.groupby('snapshot_day').apply(lambda d: pd.Series({'tgt_mean':d.future_spend_4w.mean(),'pred_mean':d[pcol].mean(),'bias':(d[pcol]-d.future_spend_4w).mean(),'mae':np.abs(d[pcol]-d.future_spend_4w).mean()}))
print(g)

feats = A.load_saved('feats_v4.parquet')
print('feats shape', feats.shape)
print('feats cols', feats.columns.tolist())
