import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days
sd = snapshot_days(); trd = sd['train']; vld = sd['validation']
tt = train_targets(); key=['household_key','snapshot_day']
base = load_saved('e014_base.parquet'); oob = load_saved('e014_gbm_oob.parquet'); stk = load_saved('e014_stack.parquet')
cmp = base[key+['gbm_pred']].merge(oob[key+['gbm_corr']], on=key)
print('gbm_pred vs gbm_corr corr:', round(cmp['gbm_pred'].corr(cmp['gbm_corr']),4))
v = cmp[cmp.snapshot_day.isin(vld)]; t = cmp[cmp.snapshot_day.isin(trd)]
print('val: pred==corr?', np.allclose(v.gbm_pred, v.gbm_corr), '| train: pred==corr?', np.allclose(t.gbm_pred, t.gbm_corr))
m = oob[key+['gbm_corr']].merge(tt, on=key)
m['ae'] = (m['gbm_corr']-m[TARGET]).abs()
g = m.groupby('snapshot_day').apply(lambda d: pd.Series({'mae': d.ae.mean(), 'corr': d.gbm_corr.corr(d[TARGET]), 'n': len(d)}))
print(g.round(3))
print('TRAIN overall gbm_corr MAE:', round(m.ae.mean(),2))
print('val gbm_corr describe:'); print(v.gbm_corr.describe().round(2))
print('train gbm_corr describe:'); print(t.gbm_corr.describe().round(2))
for nm in ['e009_analog','e009_demo_mkt','e009_spline2p']:
    d = load_saved(nm+'.parquet'); print('==',nm, d.shape); print(list(d.columns))