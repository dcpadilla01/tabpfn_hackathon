import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
base_cols = set(base.columns)
others = ["e019_true_union","e019_everything","e019_full_merged","e017_v2","e017_disc_seasonal","e016_smoothed","e013_peers","e018_timing_hazard","nf_candidates","nf_robust","nf_seasonal","nf_transforms","micro","nf_compact19","e006_catmix_mkt","e007_logratio","e010_l13fix","e011_demo","e012_dorm","e015_peers_demo_l13fix","e002_channel","e003_catmix","e004_mkt_full","e001_txhist"]
missing = {}
for n in others:
    try:
        df = A.load_saved(n+".parquet")
    except Exception as e:
        print(n, "ERR", e); continue
    miss = [c for c in df.columns if c not in base_cols and c not in ("household_key","snapshot_day")]
    if miss:
        missing[n] = miss
        print(n, df.shape, "MISSING:", miss)
print()
print("base feats:", base.shape[1]-2)