
import agent_api as api, pandas as pd, numpy as np

t = api.load_saved('e001_rfm.parquet')
print('shape', t.shape)
print('cols', list(t.columns))
print('snapshot_days', api.snapshot_days())

tt = api.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share', (tt.future_spend_4w == 0).mean())

m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
num = [c for c in t.columns if c not in ('household_key','snapshot_day') and str(m[c].dtype) in ('int64','float64','bool')]
corr = m[num + ['future_spend_4w']].corr(numeric_only=True)['future_spend_4w'].drop('future_spend_4w').sort_values()
print(corr)
