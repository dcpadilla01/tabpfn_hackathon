import agent_api as A
import pandas as pd, numpy as np

base_cols = None
for n in ["e006_composition","e007_lagseq","e009_basket","e010_decay","e011_price","dm_exp","e016_peer","e013_te_clean","cand1","e014_dm"]:
    t = A.load_saved(n + ".parquet")
    cols = list(t.columns)
    if base_cols is None:
        base_cols = list(A.load_saved("rfm28.parquet").columns)
    extra = [c for c in cols if c not in base_cols]
    print(f"== {n} {t.shape}  extra({len(extra)}):")
    print("   " + ", ".join(extra))
    print()
