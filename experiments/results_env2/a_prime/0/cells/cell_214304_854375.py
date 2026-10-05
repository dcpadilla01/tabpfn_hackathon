import pandas as pd, numpy as np
snap = agent_api.snapshot()  # capped at day 459
ct = snap.table("campaign_targets"); cp = snap.table("campaigns"); cr = snap.table("coupon_redemptions")
dm = snap.table("display_mailer"); tx = snap.table("transactions")
print("campaigns:\n", cp.head(15).to_string())
print("types:", cp["description"].value_counts().to_dict())
print("n campaign_targets rows", len(ct), "households", ct.household_key.nunique(), "campaigns", ct.campaign.nunique())
print("ct desc:\n", ct["description"].value_counts())
print("redemptions rows", len(cr), "households", cr.household_key.nunique(), "days", cr.day.min(), cr.day.max())
print("cr per hh describe:\n", cr.groupby("household_key").size().describe())
print("display_mailer rows", len(dm), "weeks", dm.week_no.min(), dm.week_no.max())
print("tx day range", tx.day.min(), tx.day.max(), "n hh", tx.household_key.nunique())
# overlap: how many eval households have redemptions/targets
hh_eval = set(tx.household_key.unique())
print("hh with targets: %d/%d" % (ct.household_key.nunique(), len(hh_eval)))
print("hh with redemptions: %d/%d" % (cr.household_key.nunique(), len(hh_eval)))
# how many transactions match display_mailer keys
sub = tx[tx.day >= 459-28]
w = (sub.day + 8)//7
key = sub.assign(week_no=w).merge(dm, on=["product_id","store_id","week_no"], how="left")
print("tx last28 rows", len(sub), "matched to dm:", key.display.notna().mean().round(3))
