
import pandas as pd, numpy as np, re
t8  = load_saved("e008_spendproc.parquet")
t12 = load_saved("e012_outcome2.parquet")
t17 = load_saved("e017_merged.parquet")

print("E008 incr cols:", [c for c in t8.columns if c.startswith(("sp_","roll","w28","proc"))][:40])
print()
print("E012 te2 cols:", [c for c in t12.columns if c.startswith("te2")])
print()
print("E017 c17 cols:", sorted([c for c in t17.columns if c.startswith("c17")]))
print()
print("E013 store/prod cols:", sorted([c for c in t17.columns if c.startswith(("st_","pr_","bk"))]))
