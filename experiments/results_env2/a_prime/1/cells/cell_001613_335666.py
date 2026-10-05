
import pandas as pd, numpy as np

rr = agent_api.load_saved('rawrec.parquet')
print("rawrec shape:", rr.shape)
print("cols:", rr.columns.tolist())
print("\nrows per snapshot_day:")
print(rr.groupby('snapshot_day').size())
print("\nNaN counts per column (sample):")
print(rr.isna().sum().sort_values(ascending=False).head(20))
print("\nhead:\n", rr.head(3))
print("\nall-NaN rows per snapshot_day:")
print(rr.groupby('snapshot_day').apply(lambda g: int(g.isna().all(axis=1).sum())))
