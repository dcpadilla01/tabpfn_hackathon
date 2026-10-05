
import agent_api, pandas as pd, numpy as np
snap = agent_api.snapshot()
tx = snap.table('transactions')
print("tx shape:", tx.shape)
sc = tx.groupby('store_id')['sales_value'].agg(['sum','count']).sort_values('sum', ascending=False)
print("n stores:", tx.store_id.nunique())
print(sc.head(15))
hs = tx.groupby('household_key')['store_id'].nunique()
print("distinct stores per hh:\n", hs.describe())
for c in ['coupon_disc','coupon_match_disc','retail_disc']:
    pos = (tx[c] != 0).mean()
    print(c, "nonzero frac:", round(pos,4), "mean:", round(tx[c].mean(),3), "min:", tx[c].min(), "max:", tx[c].max())
cr = snap.table('coupon_redemptions')
print("redemptions:", cr.shape, "hh with redemptions:", cr.household_key.nunique())
print("redemptions per hh:\n", cr.groupby('household_key').size().describe())
print("trans_time:\n", tx.trans_time.describe())
# target distribution
tt = agent_api.train_targets()
print("targets:\n", tt.future_spend_4w.describe())
print("zero frac:", (tt.future_spend_4w==0).mean())
