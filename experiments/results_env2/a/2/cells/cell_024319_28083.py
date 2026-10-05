
import agent_api as A
import pandas as pd
f3 = A.load_saved("feats_v3.parquet")
print(list(f3.columns))
fs = A.load_saved("feats_seasonal.parquet")
print(list(fs.columns))
