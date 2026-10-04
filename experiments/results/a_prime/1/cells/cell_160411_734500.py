import agent_api as A
import pandas as pd

for n in ["e013_peers.parquet","micro.parquet","nf_candidates.parquet"]:
    df = A.load_saved(n)
    print("==", n, df.shape)
    for c in df.columns:
        print("  ", c, df[c].dtype)
