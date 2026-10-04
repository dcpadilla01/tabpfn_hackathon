import agent_api as A, pandas as pd, numpy as np
v = A.snapshot()
print(v.campaigns.to_string())
red = v.coupon_redemptions
print("redemption days:", red.day.min(), red.day.max())
print(red.groupby('campaign').size())
# how many households needing rows have ever been targeted / redeemed?
hh = v.households
print("households needing rows:", len(hh))
ct = v.campaign_targets
print("targeted overlap:", hh.isin(ct.household_key).sum())
print("redeem overlap:", hh.isin(red.household_key).sum())
# transactions size
print("tx rows up to 459:", len(v.transactions))
tx = v.transactions
print("tx per household median:", tx.groupby('household_key').size().median())
# quick signal check: campaigns targeted in last 56 days vs train target
tt = A.train_targets()
print(tt.shape, tt.head())
