import agent_api as A, pandas as pd, numpy as np
ds = A.load_saved("e017_disc_seasonal.parquet")
print([c for c in ds.columns if "tenure" in c or "nspend_l" in c or "nratio_l" in c])