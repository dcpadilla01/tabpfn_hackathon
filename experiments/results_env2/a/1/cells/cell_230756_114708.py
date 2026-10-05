
import pandas as pd, numpy as np, agent_api
p5 = agent_api.load_saved("e005_preds.parquet").rename(columns={"prediction":"pred_5"})
p4 = agent_api.load_saved("e004_preds.parquet").rename(columns={"prediction":"pred_4"})
p3 = agent_api.load_saved("e003_preds.parquet").rename(columns={"prediction":"pred_3"})
m = p5.merge(p4, on=["household_key","snapshot_day"]).merge(p3, on=["household_key","snapshot_day"])
print("corr 5v4:", np.corrcoef(m.pred_5, m.pred_4)[0,1], "corr 5v3:", np.corrcoef(m.pred_5, m.pred_3)[0,1])
print("mean abs diff 5-4:", np.mean(np.abs(m.pred_5-m.pred_4)))
print(m.groupby("snapshot_day")[["pred_5","pred_4","pred_3"]].mean())

# distribution of preds vs y
tt = agent_api.train_targets()
print(tt.future_spend_4w.describe())
print(m.pred_5.describe())
