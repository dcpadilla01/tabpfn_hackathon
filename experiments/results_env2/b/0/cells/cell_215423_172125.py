import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
print("m shape:", m.shape, "cols sample:", m.columns[:6].tolist())
mu_obj = m[m.snapshot_day<431].future_spend_4w.mean()
print("mu type:", type(mu_obj), "shape:", np.shape(mu_obj))
print("snapshot_day dtype:", m.snapshot_day.dtype, "unique:", sorted(m.snapshot_day.unique())[:20])
