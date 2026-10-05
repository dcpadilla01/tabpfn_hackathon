
import agent_api, pandas as pd, numpy as np

# 1) Aggregate weekly spend seasonality over the 2 years
snap = agent_api.snapshot(459)
tx = snap.transactions
wk = tx.groupby("week_no").sales_value.sum()
print("weekly total spend: mean %.0f std %.0f cv %.2f, min wk %d (%.0f), max wk %d (%.0f)" %
      (wk.mean(), wk.std(), wk.std()/wk.mean(), wk.idxmin(), wk.min(), wk.idxmax(), wk.max()))
# print a coarse profile
prof = wk.reset_index(); prof["blk"] = (prof.week_no-1)//13
print(prof.groupby("blk").sales_value.mean().round(0))

# 2) verify lag1_spend alignment in feats_v3: spend in [s-27, s]?
f3 = agent_api.load_saved("feats_v3.parquet")
hh = f3.household_key.iloc[0]
row = f3.iloc[0]
s = int(row.snapshot_day)
h = agent_api.history(hh, as_of_day=s)
m = (h.day >= s-27) & (h.day <= s)
print("\ncheck lag1: hh", hh, "snap", s, "computed", h[m].sales_value.sum().round(2), "feat", row.lag1_spend)
m2 = (h.day >= s-55) & (h.day <= s-28)
print("check lag2: computed", h[m2].sales_value.sum().round(2), "feat", row.lag2_spend)

# 3) does a year-ago aligned window exist for most rows? tenure distribution
tt = agent_api.train_targets()
ten = f3[["household_key","snapshot_day","tenure"]]
print("\ntenure describe:"); print(ten.tenure.describe().round(0))
print("share tenure>=364:", (ten.tenure>=364).mean().round(3))
