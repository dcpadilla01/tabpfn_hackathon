import pandas as pd, numpy as np, agent_api
tt = agent_api.train_targets()
feats = agent_api.load_saved("e010_store_mix.parquet")
df = feats.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged:", df.shape)
ycol="future_spend_4w"
num = [c for c in df.columns if c not in ("household_key","snapshot_day",ycol) and pd.api.types.is_numeric_dtype(df[c])]
cor = df[num].corrwith(df[ycol]).sort_values()
print("TOP +corr:"); print(cor.tail(15).round(3).to_string())
print("TOP -corr:"); print(cor.head(8).round(3).to_string())
# target autocorrelation across snapshots (train only)
tt2 = tt.sort_values(["household_key","snapshot_day"])
tt2["lag_t"] = tt2.groupby("household_key")[ycol].shift(1)
print("\ncorr(target, target_lag1):", tt2[[ycol,"lag_t"]].corr().iloc[0,1].round(3))
# zero share
print("zero share:", (tt[ycol]==0).mean().round(3), " share<10:", (tt[ycol]<10).mean().round(3))
