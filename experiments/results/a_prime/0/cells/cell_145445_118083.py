import numpy as np, pandas as pd
s = snapshot()
tx = s.transactions
wk = (tx.day + 8)//7
prof = tx.assign(_w=wk).groupby('_w').sales_value.sum()
prof_n = tx.assign(_w=wk).groupby('_w').basket_id.nunique()
print("weeks:", prof.index.min(), prof.index.max())
print("weekly spend (first 30):")
print(prof.head(30).round(0).to_string())
print("top 12 spend weeks:")
print(prof.sort_values(ascending=False).head(12).round(0).to_string())
print("bottom 6:", prof.sort_values().head(6).round(0).values)
print("mean weekly:", round(prof.mean(),0), "std:", round(prof.std(),0))
# day-level pattern
dd = tx.groupby('day').sales_value.sum()
print("day-of-week effect (day mod 7):")
print(dd.groupby(dd.index % 7).mean().round(0).to_string())