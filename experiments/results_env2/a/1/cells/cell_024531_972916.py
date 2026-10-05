import agent_api as api
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
allF = api.load_saved('allF.parquet')
tt = api.train_targets()
m = tt.merge(allF.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'])
print("merged", m.shape)
corr = m.corr(numeric_only=True)['future_spend_4w'].drop('future_spend_4w')
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
print("top |corr|:")
print(corr.head(35).round(3))
# per-snapshot means of key features vs target mean
keys = ['spend_28','spend_84','spend_364','spend_lag1y','spend_7','spend_seas_364','spend_seas_336','seas_ok','recency','zero28','wk_zero_12','wk_mean_12']
print(allF.groupby('snapshot_day')[keys].mean().round(1))
print("\ntrain target mean by snapshot:")
print(tt.groupby('snapshot_day')['future_spend_4w'].mean().round(1))
# distribution of target and of spend_28
print("\ntarget describe:"); print(tt['future_spend_4w'].describe().round(2))
print("\nspend_28 describe:"); print(allF['spend_28'].describe().round(2))
print("\nspend_84 describe:"); print(allF['spend_84'].describe().round(2))
# how well does raw spend_28 predict? MAE of spend_28 vs target on train
mm = m.dropna(subset=['spend_28'])
print("MAE spend_28 as pred:", np.abs(mm['spend_28']-mm['future_spend_4w']).mean().round(3))
print("MAE 0.9*spend_28:", np.abs(0.9*mm['spend_28']-mm['future_spend_4w']).mean().round(3))
print("MAE spend_84*0.31:", np.abs(0.31*mm['spend_84']-mm['future_spend_4w']).mean().round(3))
# check validation rows exist in allF
val = allF[allF.snapshot_day>=459]
print("\nval rows in allF:", val.shape, "unique hh:", val.household_key.nunique())
print(val.groupby('snapshot_day').size())
