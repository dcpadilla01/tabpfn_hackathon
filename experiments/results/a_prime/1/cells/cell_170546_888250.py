
import agent_api as A
import pandas as pd, numpy as np

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w']

# Does the recent->future mapping drift across snapshots? Fit lin(spend_l1) per snapshot (in-sample)
print("In-sample linear slope/intercept of target on spend_l1, by snapshot:")
for d, g in m.groupby('snapshot_day'):
    b, a = np.polyfit(g['spend_l1'].values, g['future_spend_4w'].values, 1)
    r = g['spend_l1'].corr(g['future_spend_4w'])
    print(f"  day {d}: slope {b:.3f} intercept {a:7.1f} corr {r:.3f} n {len(g)}")

# ratio target/spend_l1 by snapshot
ratio = m['future_spend_4w']/m['spend_l1'].replace(0,np.nan)
print("\nmedian(target/spend_l1) by snapshot:")
print(m.assign(r=ratio).groupby('snapshot_day')['r'].median().round(3))
