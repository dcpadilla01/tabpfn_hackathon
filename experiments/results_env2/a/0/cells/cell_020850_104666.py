
import agent_api, pandas as pd, numpy as np

f3 = agent_api.load_saved("feats_v3.parquet")
print("feats_v3 cols:", list(f3.columns)[:40])

# per-household weekly spend seasonality (normalized by # active households)
snap = agent_api.snapshot(459)
tx = snap.transactions
g = tx.groupby(["week_no","household_key"]).sales_value.sum().reset_index()
per_hh = g.groupby("week_no").sales_value.mean()
prof = per_hh.reset_index(); prof["blk"] = (prof.week_no-1)//13
print("\nper-active-household weekly spend by 13wk block:")
print(prof.groupby("blk").sales_value.mean().round(1))
# finer: 4-week blocks
prof["blk4"] = (prof.week_no-1)//4
b4 = prof.groupby("blk4").sales_value.mean().round(1)
print("\nper-hh weekly spend by 4wk block:")
print(b4.to_string())

# lag alignment check
hh = f3.household_key.iloc[0]; s = int(f3.snapshot_day.iloc[0])
h = agent_api.history(hh, as_of_day=s)
print("\ncheck hh", hh, "snap", s,
      "lag1 calc", h[(h.day>=s-27)&(h.day<=s)].sales_value.sum().round(2),
      "feat", f3.lag1_spend.iloc[0],
      "| lag2 calc", h[(h.day>=s-55)&(h.day<=s-28)].sales_value.sum().round(2),
      "feat", f3.lag2_spend.iloc[0])
