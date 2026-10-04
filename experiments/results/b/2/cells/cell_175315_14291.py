import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e019_final.parquet')
print("shape:", t.shape)
cols = list(t.columns)
print("n cols:", len(cols))
for i in range(0, len(cols), 6):
    print(" | ".join(cols[i:i+6]))
print(t.dtypes.value_counts())
tt = A.train_targets()
print("targets:", tt.shape)
print(A.snapshot_days())