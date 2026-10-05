import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
print(df.shape, df.snapshot_day.unique())
print(df.dtypes.value_counts())
obj_cols = [c for c in df.columns if df[c].dtype==object]
print(obj_cols)
for c in obj_cols: print(c, df[c].nunique())
