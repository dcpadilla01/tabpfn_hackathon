import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

t = load_saved('e017_xsec_rank.parquet')
tt = train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]

for c in feat_cols:
    s = df[c]
    if not np.issubdtype(s.dtype, np.number) and s.dtype != bool:
        print('NONNUM', c, s.dtype)
    if np.issubdtype(s.dtype, np.number):
        if s.isna().all():
            print('ALLNAN', c)
        elif s.isna().mean() > 0.9:
            print('MOSTNAN %.2f' % s.isna().mean(), c)

# check target NaN rows
print('rows', len(df), 'nan y', int(df.future_spend_4w.isna().sum()))
print(df[df.future_spend_4w.isna()]['snapshot_day'].value_counts())
