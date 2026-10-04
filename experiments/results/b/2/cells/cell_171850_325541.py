import agent_api as A
import pandas as pd, numpy as np

for name in ['e014_gbm_oob.parquet', 'e014_stack.parquet', 'e013_denoise.parquet']:
    df = A.load_saved(name)
    print('==', name, df.shape)
    print(df.columns.tolist())
    print(df.head(3))
    print()
