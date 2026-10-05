import agent_api, pandas as pd
base = agent_api.load_saved("e006_newblock.parquet")
print(base.shape)
cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
print(len(cols))
print(cols)
