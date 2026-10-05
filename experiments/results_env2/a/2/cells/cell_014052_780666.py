import agent_api, pandas as pd, numpy as np
for name in ["feats_v4","feats_v5","feats_v6"]:
    df = agent_api.load_saved(name+".parquet")
    print("==", name, df.shape)
    print([c for c in df.columns if c not in agent_api.load_saved("feats_v3.parquet").columns])
