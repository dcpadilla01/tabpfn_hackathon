import agent_api, pandas as pd
snap = agent_api.snapshot()
print(snap.campaigns.sort_values('start_day'))
print("\ndepartments:")
print(snap.products['department'].value_counts().head(15))
print("n departments:", snap.products['department'].nunique())
print("\nbrand:", snap.products['brand'].value_counts(dropna=False))
tx = snap.transactions
print("\ntx shape:", tx.shape, "max day:", tx.day.max())
print(tx[['coupon_disc','retail_disc','coupon_match_disc']].describe())
print("\nredemptions per household:")
r = snap.coupon_redemptions.groupby('household_key').size()
print(r.describe())
