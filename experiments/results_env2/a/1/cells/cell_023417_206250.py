import pandas as pd, numpy as np
pd.set_option('display.width', 250)

allF = agent_api.load_saved('allF.parquet')
print('allF cols (%d):' % allF.shape[1])
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day')]
print(feat_cols)
print(allF[feat_cols].dtypes.value_counts())
# NaN structure by snapshot for a few cols
sub = allF.groupby('snapshot_day')[['spend_364','spend_lag1y','spend_seas_364','seas_ok','sp28_yag','wk_52']].apply(lambda d: d.isna().mean().round(3))
print(sub)

tt = agent_api.train_targets()
allF2 = allF.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = allF2[allF2.snapshot_day<=431]
print('\nmean target vs mean spend_28 by snapshot (trend check):')
g = tr.groupby('snapshot_day').agg(y=('future_spend_4w','mean'), s28=('spend_28','mean'), s84=('spend_84','mean'), n=('future_spend_4w','size'))
g['y_over_s28'] = g.y/g.s28
print(g.round(3))