
import pandas as pd, numpy as np, agent_api

for name in ["e004_features","e005_newfeats","e005_preds","e004_preds","e003_preds"]:
    df = agent_api.load_saved(name+".parquet")
    print(name, df.shape)
    print(list(df.columns)[:40])
    print(df.head(3))
    print("---")
