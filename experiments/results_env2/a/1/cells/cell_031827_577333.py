
import agent_api, numpy as np, pandas as pd

tt = agent_api.train_targets()
print("train rows:", tt.shape)
print(tt['future_spend_4w'].describe())
print("quantiles:", np.percentile(tt['future_spend_4w'], [50,75,90,95,99]))

# load saved prediction tables and check correlation / per-day bias on train rows
names = ["e005_preds","e004_preds","e003_preds","e011_preds","e007_preds","e009_preds","e008_preds","e010_preds","e012_preds","e014_preds","e001_preds","e002_preds"]
for n in names:
    try:
        df = agent_api.load_saved(n + ".parquet")
        m = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
        p = m['prediction']
        t = m['future_spend_4w']
        print(n, "cols:", list(df.columns), "mae:", np.mean(np.abs(p-t)).round(3),
              "bias:", np.mean(p-t).round(3), "corr:", np.corrcoef(p,t)[0,1].round(3))
    except Exception as e:
        print(n, "ERR", type(e).__name__, e)
