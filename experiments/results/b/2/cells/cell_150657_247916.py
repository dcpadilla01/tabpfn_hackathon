
import pandas as pd, numpy as np

e7 = load_saved('e007_log.parquet')
print('e7 cols:', len(e7.columns)-2)
# which features are NOT log1p-transformed
sk = e7.drop(columns=['household_key','snapshot_day']).skew()
print('skew>3 (raw-scale left):')
print(sk[sk.abs()>3].round(1).to_string())
print('\nNaN counts (top):')
na = e7.isna().sum()
print(na[na>0].sort_values(ascending=False).head(12).to_string())
