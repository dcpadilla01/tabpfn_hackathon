import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]
na = m[feat].isna().mean().sort_values(ascending=False)
print(na[na>0])
