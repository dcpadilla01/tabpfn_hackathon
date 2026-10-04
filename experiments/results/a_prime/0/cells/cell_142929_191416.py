import agent_api as A
import pandas as pd, numpy as np

print("snapshot days:", A.snapshot_days())
tt = A.train_targets()
print("train targets:", tt.shape)
print(tt.future_spend_4w.describe())
print("zero share:", (tt.future_spend_4w == 0).mean())

v = A.snapshot()
print("tx", v.transactions.shape)
print("demo", v.demographics.shape)
print("camp", v.campaigns.shape)
print("ctgt", v.campaign_targets.shape)
print("cred", v.coupon_redemptions.shape)
print("coup", v.coupons.shape)
print("dm", v.display_mailer.shape)
print("prod", v.products.shape)

tx = v.transactions
print("tx day range:", tx.day.min(), tx.day.max())
print("nunique hh in tx:", tx.household_key.nunique())
print("KEYS:", A.KEYS, "TARGET:", A.TARGET)
print(tx.head(3))
