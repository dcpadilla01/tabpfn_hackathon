import agent_api as A
import pandas as pd, numpy as np

e015 = A.load_saved('e015_base.parquet')
print("e015 shape:", e015.shape)
print("cols:", list(e015.columns)[:130])
print()
print("saved tables:", A.describe_tables() if hasattr(A,'describe_tables') else '')