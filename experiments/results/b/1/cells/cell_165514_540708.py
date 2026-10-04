import agent_api as A
import pandas as pd, numpy as np

base = A.load_saved("e015_base.parquet")
print("e015_base cols:")
for c in base.columns: print("  ", c)