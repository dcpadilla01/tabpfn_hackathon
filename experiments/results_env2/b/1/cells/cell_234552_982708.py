import agent_api, pandas as pd, numpy as np

e8 = agent_api.load_saved("e008_level_shape.parquet")
tt = agent_api.train_targets()
tr = tt.merge(e8, on=["household_key","snapshot_day"], how="left")
y = tr['future_spend_4w'].values

# check dtypes
num_cols = [c for c in e8.columns if c not in ("household_key","snapshot_day") and pd.api.types.is_numeric_dtype(e8[c])]
cat_cols = [c for c in e8.columns if c not in ("household_key","snapshot_day") and not pd.api.types.is_numeric_dtype(e8[c])]
print("numeric:", len(num_cols), "categorical:", len(cat_cols), cat_cols)

# scales of key features
key = ['sp28','sp56','sp84','sp168','sp364','z_log_sp28','z_med4w_hist','z_mean_week_spend_all','days_since_last','tenure','wksp_mean','splag1y']
print(e8[key].describe().T[['mean','std','min','max']])

# global median predictor MAE on all train rows
print("global median mae:", np.mean(np.abs(y-np.median(y))))
# corr of key feats with target
for c in key:
    print(c, round(np.corrcoef(tr[c].fillna(0), y)[0,1],3))
