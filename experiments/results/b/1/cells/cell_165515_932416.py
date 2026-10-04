import agent_api as A
import pandas as pd, numpy as np
for nm in ["cand_new","e006_seq_gaps","e012_full","e004_long_hist","e007_new"]:
    df = A.load_saved(nm + ".parquet")
    print(f"--- {nm} ({df.shape[1]-3} feats) ---")
    print([c for c in df.columns if c not in ("household_key","snapshot_day")])