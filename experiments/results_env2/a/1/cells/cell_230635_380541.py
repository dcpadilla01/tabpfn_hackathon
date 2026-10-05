
import pandas as pd, numpy as np, agent_api

feats = agent_api.load_saved("e004_features.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"])
print(df.shape)

y = df.future_spend_4w.values

# how predictive is spend_28 alone (MAE-optimal affine)?
s28 = df.spend_28.values if "spend_28" in df.columns else None
print([c for c in df.columns if "spend" in c][:20])
