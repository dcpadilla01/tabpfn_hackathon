
import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
f4 = agent_api.load_saved("feats_v4.parquet")
p = agent_api.load_saved("pred_e013.parquet")
print(tt.dtypes, "\n", f4.dtypes.head(3), "\n", p.dtypes)
print(tt.household_key.dtype, f4.household_key.dtype, p.household_key.dtype)
m = f4.merge(p, on=["household_key","snapshot_day"])
print("f4+pred", m.shape)
m2 = f4.merge(tt, on=["household_key","snapshot_day"])
print("f4+tt", m2.shape)
m3 = f4.astype({"household_key":"int64","snapshot_day":"int64"}).merge(
     tt.astype({"household_key":"int64","snapshot_day":"int64"}), on=["household_key","snapshot_day"])
print("cast f4+tt", m3.shape)
