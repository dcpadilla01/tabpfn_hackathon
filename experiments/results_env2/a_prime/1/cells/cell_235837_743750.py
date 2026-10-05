
import pandas as pd, numpy as np
r = agent_api.load_saved('rawrec.parquet')
print(r.shape)
print(r.columns.tolist())
print(r.head(3))
