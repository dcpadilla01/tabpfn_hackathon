import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
e5 = A.load_saved("e005_preds.parquet")
# per-snapshot-day target stats (train)
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median",lambda s:(s==0).mean(),"count"])
g.columns=["mean","median","zerofrac","n"]; print("TRAIN targets by day\n", g.round(1))
# e5 predictions by snapshot day (validation only in preds? check)
print("pred days:", e5.snapshot_day.unique())
pv = e5.groupby("snapshot_day").prediction.agg(["mean","median",lambda s:(s<=0).mean(),"count"])
pv.columns=["mean","median","zerofrac","n"]; print("E5 preds by day\n", pv.round(1))
# blend check: correlation of e3/e4/e5
m = e5.merge(A.load_saved("e004_preds.parquet"), on=["household_key","snapshot_day"], suffixes=("_e5","_e4")).merge(A.load_saved("e003_preds.parquet").rename(columns={"prediction":"pred_e3"}), on=["household_key","snapshot_day"])
print(m[["prediction_e5","prediction_e4","pred_e3"]].corr().round(4))
