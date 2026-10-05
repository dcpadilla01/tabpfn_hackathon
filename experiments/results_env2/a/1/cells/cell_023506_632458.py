import pandas as pd, numpy as np
pd.set_option('display.width', 250)
tt = agent_api.train_targets()
e5 = agent_api.load_saved('e005_preds.parquet')
print(tt.dtypes, '\n', e5.dtypes)
print(tt['snapshot_day'].unique()[:5], e5['snapshot_day'].unique()[:5])
print(tt['household_key'].dtype, e5['household_key'].dtype, tt['household_key'].iloc[0], e5['household_key'].iloc[0])
m = tt.merge(e5, on=['household_key','snapshot_day'])
print('merged', m.shape)
from sklearn.metrics import mean_absolute_error
print('MAE e005:', mean_absolute_error(m['future_spend_4w'], m['prediction']))
print(m.groupby('snapshot_day').apply(lambda d: mean_absolute_error(d['future_spend_4w'], d['prediction']), include_groups=False).round(2))
d = m['prediction']-m['future_spend_4w']
print('mean bias', d.mean().round(3), 'median bias', d.median().round(3))
qb = pd.qcut(m['future_spend_4w'], 10, duplicates='drop')
print(m.groupby(qb, observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g['future_spend_4w'].mean(),'p':g['prediction'].mean(),'bias':(g['prediction']-g['future_spend_4w']).mean()}), include_groups=False).round(2))
# check repro_e5 vs e005_preds
r5 = agent_api.load_saved('repro_e5.parquet')
mm = e5.merge(r5, on=['household_key','snapshot_day'])
print('repro check: corr p13 vs prediction', np.corrcoef(mm['p13'], mm['prediction'])[0,1].round(4), 'mean abs diff', (mm['p13']-mm['prediction']).abs().mean().round(3))
print('p11 vs p13 corr', np.corrcoef(mm['p11'], mm['p13'])[0,1].round(4), 'mean abs diff', (mm['p11']-mm['p13']).abs().mean().round(3))