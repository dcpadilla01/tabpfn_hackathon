import agent_api as A, pandas as pd
r = A.load_saved('rhythm_v1.parquet')
print(type(r), r.shape)
print(r.index.names, r.columns[:5].tolist())
print(r.head(2).to_string())
base = A.load_saved('e009_demo.parquet')
print('base', base.shape, base.index.names, base.columns[:3].tolist())