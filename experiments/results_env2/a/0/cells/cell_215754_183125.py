import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, sklearn
print("xgb", xgb.__version__, "sklearn", sklearn.__version__)
print(A.snapshot_days())
f4 = A.load_saved("feats_v3.parquet")
print(f4.shape); print(f4.columns.tolist()[:40])
t = A.train_targets(); print(t.shape); print(t.future_spend_4w.describe())
