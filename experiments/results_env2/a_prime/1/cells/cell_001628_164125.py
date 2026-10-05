
import pandas as pd, numpy as np
rr = agent_api.load_saved('rawrec.parquet')
print(rr[['g','sp28','sp56','ew','wk28']].head(8))
print("\nunique g:", rr['g'].nunique(), " rows:", len(rr))
# check if g encodes household|day
sample = rr['g'].astype(str).head(5).tolist()
print("g samples:", sample)
# how many rows per household?
rr['hh'] = rr['g'].astype(str).str.split('|').str[0]
print("\nrows per hh (describe):", rr.groupby('hh').size().describe())
# NaN structure
print("\nNaN cols:\n", rr.isna().sum().sort_values(ascending=False).head(8))
print("\nany-nan rows:", int(rr.isna().any(axis=1).sum()))
