import agent_api, numpy as np, pandas as pd
print(agent_api.snapshot_days())
tt = agent_api.train_targets()
print("targets:", tt.shape)
print(tt.future_spend_4w.describe())
snap = agent_api.snapshot()
tr = snap.transactions
print("transactions:", tr.shape)
print(tr.sales_value.describe())
for c in ['coupon_disc','coupon_match_disc','retail_disc','quantity','day']:
    print(c, tr[c].describe().round(2).to_dict())
hh = snap.households
print("households at 459:", len(hh))
print(tr.head(3))
