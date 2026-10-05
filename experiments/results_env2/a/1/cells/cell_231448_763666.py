import agent_api as A, pandas as pd, numpy as np
for name in ["e005_preds","e005_newfeats","e004_features","e004_new","e002_features","e004_preds","e003_preds"]:
    try:
        df = A.load_saved(name+".parquet")
        print(name, df.shape, list(df.columns)[:12], "..." if df.shape[1]>12 else "")
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
tt = A.train_targets()
print("targets", tt.shape, tt.columns.tolist())
print("zero frac", (tt.future_spend_4w==0).mean(), "mean", tt.future_spend_4w.mean(), "median", tt.future_spend_4w.median(), "p90", tt.future_spend_4w.quantile(.9))
print(A.snapshot_days())
