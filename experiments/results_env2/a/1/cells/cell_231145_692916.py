
import pandas as pd, numpy as np, agent_api
feats = agent_api.load_saved("e004_features.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"])

# household-level consistency: same household across snapshots
g = df.groupby("household_key").future_spend_4w.agg(["mean","std","count"])
print("households with >=2 snapshots:", (g["count"]>=2).sum())
print("mean within-household std:", g.loc[g["count"]>=2,"std"].mean())
print("mean within-household std / overall std:", g.loc[g["count"]>=2,"std"].mean()/df.future_spend_4w.std())
# ICC-like: between-household variance share
m = df.future_spend_4w.mean()
bw = df.groupby("household_key").future_spend_4w.mean().var()
tot = df.future_spend_4w.var()
print("between-household var share:", bw/tot)
