
import agent_api, pandas as pd, numpy as np

feats = agent_api.load_saved("feats_v4.parquet")
print("feats_v4:", feats.shape)
print(list(feats.columns))

tt = agent_api.train_targets()
print("\ntargets:", tt.shape, sorted(tt.snapshot_day.unique()))
m = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
print("\nper-snapshot target stats (train):")
print(m[m.future_spend_4w.notna()].groupby("snapshot_day").future_spend_4w.agg(["count","mean","median"]).round(1))

print("\nNaN counts (nonzero):")
nn = feats.isna().sum()
print(nn[nn>0])

# saved validation predictions
names = ["pred_e004","pred_e005","pred_e007","pred_e008","pred_e011","pred_e012","pred_e013","pred_e014","pred_e015","pred_e017"]
base = agent_api.load_saved("pred_e013.parquet")
print("\npred_e013:", base.shape, sorted(base.snapshot_day.unique()))
M = base[["household_key","snapshot_day"]].copy()
for nm in names:
    p = agent_api.load_saved(nm + ".parquet")
    M = M.merge(p[["household_key","snapshot_day","prediction"]].rename(columns={"prediction":nm}), on=["household_key","snapshot_day"])
P = M.drop(columns=["household_key","snapshot_day"])
print("\npred corr:")
print(P.corr().round(3).mean(axis=1).round(4))
print("\npred means per snapshot (e013):")
print(base.groupby("snapshot_day").prediction.agg(["count","mean","median"]).round(1))
