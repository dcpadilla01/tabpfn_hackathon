import numpy as np, pandas as pd
e16 = load_saved("e018_timing_hazard.parquet")  # E016 table
e17 = load_saved("e019_everything.parquet")     # E017 table
c16, c17 = set(e16.columns), set(e17.columns)
print("in E016 not E017:", sorted(c16-c17))
print("in E017 not E016:", sorted(c17-c16))
print()
dm = snapshot().display_mailer
print(dm.shape); print(dm.head(8))
print(dm['display'].value_counts().head())
print(dm['mailer'].value_counts().head())