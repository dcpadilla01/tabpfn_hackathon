
import pandas as pd, numpy as np, agent_api

names = ["pred_e013","pred_e014","pred_e011","pred_e007","pred_e005","pred_e004"]
preds = {}
for n in names:
    try:
        df = agent_api.load_saved(n + ".parquet")
        preds[n] = df
        print(n, df.shape, sorted(df.snapshot_day.unique()))
    except Exception as e:
        print(n, "ERR", type(e).__name__, e)

tt = agent_api.train_targets()
print("train_targets", tt.shape, sorted(tt.snapshot_day.unique()))

# validation rows per pred
val_days = agent_api.snapshot_days()["validation"]
base = preds["pred_e013"]
val = base[base.snapshot_day.isin(val_days)]
print("val rows", val.shape)

# pairwise correlation of predictions on validation rows
m = val[["household_key","snapshot_day","prediction"]].rename(columns={"prediction":"e013"})
for n in names[1:]:
    if n in preds:
        v = preds[n][preds[n].snapshot_day.isin(val_days)][["household_key","snapshot_day","prediction"]].rename(columns={"prediction":n})
        m = m.merge(v, on=["household_key","snapshot_day"], how="inner")
print(m.shape)
print(m.drop(columns=["household_key","snapshot_day"]).corr().round(4))

# train-row MAE of each (in-sample reference)
for n in names:
    if n in preds:
        d = preds[n].merge(tt, on=["household_key","snapshot_day"])
        if len(d):
            print(n, "train MAE", np.abs(d.prediction - d.future_spend_4w).mean().round(3), "rows", len(d))
