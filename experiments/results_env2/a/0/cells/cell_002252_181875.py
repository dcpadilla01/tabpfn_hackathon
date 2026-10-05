import agent_api as A, pandas as pd, numpy as np
feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
p13 = A.load_saved('pred_e013.parquet')
val = feats[feats.snapshot_day.isin([459,487,515,543])][['household_key','snapshot_day']]
# error analysis on TRAIN rows using in-sample? Not available. Instead analyze structure of val predictions vs train targets.
valp = val.merge(p13, on=['household_key','snapshot_day'])
print('val pred describe:'); print(valp.prediction.describe())
print('zero preds frac:', (valp.prediction<1).mean())
# train target quantiles
print('train target quantiles:', np.percentile(tt.future_spend_4w,[10,25,50,75,90,95,99]))
# how much MAE would perfect-constant-median give?
med = tt.future_spend_4w.median()
print('median-only MAE on train:', np.abs(tt.future_spend_4w-med).mean())
# distribution of train targets by spend_28 bucket (proxy for where errors live)
f = feats.merge(tt, on=['household_key','snapshot_day'])
f['b'] = pd.cut(f.spend_28, [-1,0,25,75,150,300,1e9])
print(f.groupby('b', observed=True).agg(n=('future_spend_4w','size'), mean=('future_spend_4w','mean'), med=('future_spend_4w','median'), zero=('future_spend_4w', lambda s:(s==0).mean())))
# zero-target households: what do their history features look like?
z = f[f.future_spend_4w==0]
print('zero-target n=', len(z))
print(z[['spend_28','spend_84','days_since_last','tenure']].describe().loc[['mean','50%']])
nz = f[f.future_spend_4w>0]
print(nz[['spend_28','spend_84','days_since_last','tenure']].describe().loc[['mean','50%']])
