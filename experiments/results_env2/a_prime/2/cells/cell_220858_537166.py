import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e005_decay_gapcv.parquet')
print(df.dtypes.value_counts())
print(df.isna().sum().sort_values(ascending=False).head(10))
print(df[df.columns[:5]].head(3))