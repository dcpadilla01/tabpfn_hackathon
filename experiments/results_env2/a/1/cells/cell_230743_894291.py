
import pandas as pd, numpy as np, agent_api
p5 = agent_api.load_saved("e005_preds.parquet")
p4 = agent_api.load_saved("e004_preds.parquet")
p3 = agent_api.load_saved("e003_preds.parquet")
m = p5.merge(p4, on=["household_key","snapshot_day"], suffixes=("_5","_4")).merge(p3, on=["household_key","snapshot_day"])
m = m.rename(columns={"prediction":"pred_3"})
print(m.describe())
print("corr 5 vs 4:", np.corrcoef(m.pred_5, m.pred_4)[0,1])
print("mean abs diff 5-4:", np.mean(np.abs(m.pred_5-m.pred_4)))
print("mean abs diff 5-3:", np.mean(np.abs(m.pred_5-m.pred_3)))

# per-snapshot means
print(m.groupby("snapshot_day")[["pred_5","pred_4","pred_3"]].mean())
