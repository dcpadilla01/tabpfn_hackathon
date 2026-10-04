import agent_api as api
import pandas as pd, numpy as np

v = api.snapshot(95)
print("households:", type(v.households), len(v.households))
print(v.households.head() if hasattr(v.households,'head') else v.households[:5])
print("day:", v.day, "week:", v.week)
tx = v.transactions
print("tx:", tx.shape, "maxday", tx.day.max())
print(tx.head(2))
print("demo:", v.demographics.shape)
print("ct:", v.campaign_targets.shape, "red:", v.coupon_redemptions.shape, "coup:", v.coupons.shape)
print("dm:", v.display_mailer.shape, "camp:", v.campaigns.shape)
print("prod:", v.products.shape)
