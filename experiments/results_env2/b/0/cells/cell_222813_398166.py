import agent_api as api, pandas as pd, numpy as np
base = api.load_saved("e006_newblock.parquet")
newf = api.load_saved("e009_campaign_dynamics.parquet")
m = base.merge(newf, on=["household_key","snapshot_day"], how="left")
print(m.shape, "dup:", m.duplicated(["household_key","snapshot_day"]).sum())
print(m[["c_active_n","c_act_TypeA","c_days_to_end","c_started_28"]].isna().mean().round(3).to_dict())
path = api.save_table(m, "e009_full")
print(path)
