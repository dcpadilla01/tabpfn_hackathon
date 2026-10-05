import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print(len(feat_cols))
sub = allF.groupby('snapshot_day')[['spend_364','spend_lag1y','spend_seas_364','seas_ok','s1','s2','s3','s4']].apply(lambda d: d.isna().mean().round(3))
print(sub)
tt = agent_api.train_targets()
allF2 = allF.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = allF2[allF2.snapshot_day<=431]
g = tr.groupby('snapshot_day').agg(y=('future_spend_4w','mean'), s28=('spend_28','mean'), s84=('spend_84','mean'), n=('future_spend_4w','size'))
g['y_over_s28'] = g.y/g.s28
print(g.round(3))
from sklearn.metrics import mean_absolute_error
print('MAE spend_28 as pred:', mean_absolute_error(tr['future_spend_4w'], tr['spend_28']))
print('MAE spend_84/3:', mean_absolute_error(tr['future_spend_4w'], tr['spend_84']/3))
print('MAE spend_364/13:', mean_absolute_error(tr['future_spend_4w'], tr['spend_364']/13))
print('MAE max(spend_28, s84/3):', mean_absolute_error(tr['future_spend_4w'], np.maximum(tr['spend_28'], tr['spend_84']/3)))
print('MAE blend .5:', mean_absolute_error(tr['future_spend_4w'], 0.5*tr['spend_28']+0.5*tr['spend_84']/3))