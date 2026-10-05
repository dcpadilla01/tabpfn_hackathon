import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
y = tr['future_spend_4w']
from sklearn.metrics import mean_absolute_error
print('MAE spend_364/13:', mean_absolute_error(y, tr['spend_364'].fillna(0)/13))
print('MAE max(spend_28, s84/3):', mean_absolute_error(y, np.maximum(tr['spend_28'], tr['spend_84'].fillna(0)/3)))
print('MAE blend .5:', mean_absolute_error(y, 0.5*tr['spend_28']+0.5*tr['spend_84'].fillna(0)/3))
print('\nper-snapshot MAE of spend_28:')
print(tr.groupby('snapshot_day').apply(lambda d: mean_absolute_error(d['future_spend_4w'], d['spend_28']), include_groups=False).round(2))
# NaN rates of new features at late snapshots
sub = allF.groupby('snapshot_day')[['s1','s2','s3','s4','seq_mean','ratio_s1_s3','gap_mean','wk_inact_streak','wk_best_streak','dsp28_GROC','drec_GROC','dept28_tot']].apply(lambda d: d.isna().mean().round(3))
print('\nNaN rates:'); print(sub.loc[[95,431,459,487,515,543]])
# how many rows per snapshot in allF
print('\nrows per snapshot:'); print(allF.snapshot_day.value_counts().sort_index())
# check e005 preds per snapshot MAE
e5 = agent_api.load_saved('e005_preds.parquet')
tt = agent_api.train_targets()
val = tt[tt.snapshot_day>431].merge(e5, on=['household_key','snapshot_day'])
print('\ne005 per-snapshot MAE:')
print(val.groupby('snapshot_day').apply(lambda d: mean_absolute_error(d['future_spend_4w'], d['prediction']), include_groups=False).round(2))
print('overall', mean_absolute_error(val['future_spend_4w'], val['prediction']).round(3))
# bias
d = val['prediction']-val['future_spend_4w']
print('mean bias', d.mean().round(3), 'median bias', d.median().round(3))
qb = pd.qcut(val['future_spend_4w'], 10, duplicates='drop')
print(val.groupby(qb, observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g['future_spend_4w'].mean(),'p':g['prediction'].mean(),'bias':(g['prediction']-g['future_spend_4w']).mean()}), include_groups=False).round(2))