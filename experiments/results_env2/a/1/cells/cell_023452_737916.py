import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
g = tr.groupby('snapshot_day').agg(y=('future_spend_4w','mean'), s28=('spend_28','mean'), s84=('spend_84','mean'), n=('future_spend_4w','size'))
g['y_over_s28'] = g.y/g.s28
print(g.round(3))
from sklearn.metrics import mean_absolute_error
y = tr['future_spend_4w']
print('MAE spend_28 as pred:', mean_absolute_error(y, tr['spend_28']))
print('MAE spend_84/3:', mean_absolute_error(y, tr['spend_84']/3))
print('MAE spend_364/13:', mean_absolute_error(y, tr['spend_364']/13))
print('MAE max(spend_28, s84/3):', mean_absolute_error(y, np.maximum(tr['spend_28'], tr['spend_84']/3)))
print('MAE blend .5:', mean_absolute_error(y, 0.5*tr['spend_28']+0.5*tr['spend_84']/3))
# per-snapshot MAE of spend_28
print('\nper-snapshot MAE of spend_28:')
print(tr.groupby('snapshot_day').apply(lambda d: mean_absolute_error(d['future_spend_4w'], d['spend_28']), include_groups=False).round(2))
# check NaN of s1..s4 etc at late snapshots
sub = allF.groupby('snapshot_day')[['s1','s2','s3','s4','seq_mean','ratio_s1_s3','gap_mean','wk_inact_streak','wk_best_streak','dsp28_GROC']].apply(lambda d: d.isna().mean().round(3))
print('\nNaN rates late snapshots:'); print(sub.loc[[403,431,459,487,515,543]])