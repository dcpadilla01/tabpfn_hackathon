import agent_api, pandas as pd
v3 = agent_api.load_saved("feats_v3.parquet")
seas = agent_api.load_saved("feats_seasonal.parquet")
print(v3.shape, seas.shape, seas.columns[:3].tolist())
df = v3.merge(seas.drop(columns=["snapshot_day"]), on="household_key")
print(df.shape)
print("match:", (df.snapshot_day_x == df.snapshot_day_y).all() if "snapshot_day_y" in df.columns else "n/a")
print([c for c in df.columns if "snapshot" in c])
