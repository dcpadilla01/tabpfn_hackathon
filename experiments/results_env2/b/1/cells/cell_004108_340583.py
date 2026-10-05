import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t17 = agent_api.load_saved('e017_grand.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt  = agent_api.train_targets()
print('t17', t17.shape, 't16', t16.shape)
pool = t17.merge(t16, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
print('pool', pool.shape)
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('df', df.shape)

# column types
num_cols, cat_cols = [], []
for c in df.columns:
    if c in ('household_key','snapshot_day','future_spend_4w'): continue
    if df[c].dtype.kind in 'ifb': num_cols.append(c)
    else: cat_cols.append(c)
print('num', len(num_cols), 'cat', cat_cols)
for c in cat_cols: print(c, df[c].nunique(), df[c].dtype)
