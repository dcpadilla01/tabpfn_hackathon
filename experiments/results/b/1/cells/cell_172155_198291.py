import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
print("tgt:", tgt.shape, tgt.snapshot_day.unique(), tgt.y.mean())
df = A.load_saved('e009_macro.parquet')
print("df:", df.shape, sorted(df.snapshot_day.unique()))
d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
print("d:", d.shape, "y na:", d.y.isna().mean())
vam = d[d.snapshot_day.isin([347,375,403,431])]
print("vam:", len(vam), "y mean:", vam.y.mean(), "y na:", vam.y.isna().mean())
trm = d[d.snapshot_day.isin([95,123,151,179,207,235,263,291,319])]
print("trm:", len(trm), "y mean:", trm.y.mean())
print(d.dtypes.head(8))