import agent_api as A
v = A.snapshot()
print("== campaigns", v.campaigns.shape); print(v.campaigns.head(3).to_string())
print("== campaign_targets", v.campaign_targets.shape); print(v.campaign_targets.head(3).to_string())
print("== coupon_redemptions", v.coupon_redemptions.shape); print(v.coupon_redemptions.head(3).to_string())
print("== display_mailer", v.display_mailer.shape); print(v.display_mailer.head(3).to_string())
print(v.campaign_targets.description.value_counts())
print(v.display_mailer.display.value_counts().head(10))
print(v.display_mailer.mailer.value_counts().head(10))
dm = v.display_mailer
print("dm stores:", dm.store_id.nunique(), "weeks:", dm.week_no.nunique())
red = v.coupon_redemptions
print("redemptions households:", red.household_key.nunique(), "rows:", len(red))
ct = v.campaign_targets
print("targeted households:", ct.household_key.nunique(), "rows:", len(ct))
