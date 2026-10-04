
import agent_api as A, pandas as pd, numpy as np
names = ["e018_union_full","e019_true_union","e017_v2","e016_smoothed","e013_peers","e018_timing_hazard","nf_candidates","nf_robust","nf_seasonal","nf_transforms","micro","nf_compact19"]
for n in names:
    try:
        df = A.load_saved(n+".parquet")
        print(n, df.shape)
        print(list(df.columns))
        print("---")
    except Exception as e:
        print(n, "ERR", e)
