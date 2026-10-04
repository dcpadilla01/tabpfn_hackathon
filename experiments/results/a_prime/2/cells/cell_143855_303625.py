import agent_api as A, pandas as pd, numpy as np
for p in ["temporal_structure.parquet","recency_agg.parquet","dept_mix_recency.parquet"]:
    df = A.load_saved(p)
    print(p, df.shape)
    print(list(df.columns)[:40])
tt = A.train_targets()
print(tt.shape); print(tt.head(3)); print(tt.future_spend_4w.describe())
