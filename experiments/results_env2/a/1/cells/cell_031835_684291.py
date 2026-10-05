
import agent_api, numpy as np, pandas as pd
allF = agent_api.load_saved("allF.parquet")
print("allF:", allF.shape)
print(list(allF.columns))
print(allF['snapshot_day'].value_counts().sort_index())
# check other saved tables
for n in ["e005_newfeats","lagfeats","lagfeats2","f_weekly","camp_feats","repro_e5"]:
    try:
        d = agent_api.load_saved(n + ".parquet")
        print(n, d.shape, list(d.columns)[:12])
    except Exception as e:
        print(n, "ERR", e)
