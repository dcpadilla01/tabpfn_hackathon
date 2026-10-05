v = agent_api.snapshot(459)
ct = v.table("campaign_targets")
cr = v.table("coupon_redemptions")
cp = v.table("coupons")
tx = v.table("transactions")
print("targets rows", len(ct), "hh", ct.household_key.nunique())
print("redemptions rows", len(cr), "hh", cr.household_key.nunique(), "max day", cr.day.max())
print("coupons rows", len(cp))
print("tx max day", tx.day.max(), "hh in tx", tx.household_key.nunique())
print(ct.head(3))
print(cr.groupby('campaign').size())
# how many target households are in the snapshot households set?
hh = v.households
print("snapshot hh", len(hh), "overlap targets", ct.household_key.isin(hh).sum(), "overlap redeemers", cr.household_key.isin(hh).sum())