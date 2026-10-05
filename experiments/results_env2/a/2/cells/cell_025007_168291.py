
import agent_api as A
import pandas as pd, numpy as np

# check views at arbitrary days and raw table columns at 459
v400 = A.snapshot(400)
tx = v400.table("transactions")
print("day400 tx max day:", tx.day.max(), "shape", tx.shape)
cr = v400.table("coupon_redemptions"); print("cr max day", cr.day.max())
v459 = A.snapshot()
c = v459.table("campaigns"); print(c.shape); print(c.head(3)); print("start_day max", c.start_day.max())
ct = v459.table("campaign_targets"); print("campaign_targets", ct.shape, ct.description.nunique())
dm = v459.table("display_mailer"); print("display_mailer", dm.shape, "week max", dm.week_no.max())
print(A.snapshot_days())
