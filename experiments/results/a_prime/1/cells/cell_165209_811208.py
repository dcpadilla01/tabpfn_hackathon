import numpy as np, pandas as pd
v = snapshot()
camp = v.campaigns; ct = v.campaign_targets
print(camp.shape, ct.shape)
print(camp.head())
print(camp['description'].value_counts())
print("campaign day ranges: start", camp.start_day.min(), camp.start_day.max(), "end", camp.end_day.min(), camp.end_day.max())
print(ct['description'].value_counts())
print("n households targeted:", ct.household_key.nunique())
# how many campaigns overlap a hypothetical next-28d window at day 459?
for sd in [95, 459]:
    act = camp[(camp.start_day <= sd+28) & (camp.end_day >= sd+1)]
    t = ct[ct.campaign.isin(act.campaign)]
    print(f"snapshot {sd}: campaigns overlapping next28={len(act)}, targeted hh-campaign pairs={len(t)}, distinct hh={t.household_key.nunique()}")
# redemption stats
cr = v.coupon_redemptions
print("redemptions", cr.shape, "hh", cr.household_key.nunique(), "day max", cr.day.max())
print(cr.head(3))