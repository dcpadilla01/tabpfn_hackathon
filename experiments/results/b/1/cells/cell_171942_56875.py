import agent_api as A
import pandas as pd, numpy as np

# Compute targets for train snapshots myself (windows end <= 459, visible at snapshot(459))
tt = A.train_targets()
print(tt.shape, tt.head())
snap = A.snapshot(459)
tr = snap.transactions[['household_key','day','sales_value']]
# future window spend for each train snapshot
days = A.snapshot_days()['train']
sp = tr.groupby('household_key')['sales_value'].sum()
# build window sums
g = tr.sort_values('day')
mine = {}
for d in days:
    w = tr[(tr.day > d) & (tr.day <= d+28)].groupby('household_key')['sales_value'].sum()
    mine[d] = w
mm = pd.concat([mine[d].rename(d) for d in days], axis=1)
mm = mm.stack().rename('y').reset_index().rename(columns={'level_1':'snapshot_day'})
mm['snapshot_day'] = mm['snapshot_day'].astype(int)
chk = tt.merge(mm, on=['household_key','snapshot_day'], how='left')
print("match rate:", (chk.future_spend_4w.round(2)==chk.y.round(2)).mean(), "maxdiff:", (chk.future_spend_4w-chk.y).abs().max())
print("train rows:", len(tt), "y mean/std:", tt.future_spend_4w.mean(), tt.future_spend_4w.std())
print("snapdays:", A.snapshot_days())