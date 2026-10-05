import agent_api as A
import pandas as pd, numpy as np

for name in ["pred_seasonal","pred_e008","pred_e009","pred_e010_blend","pred_e006","oof_e008","oof_harness"]:
    try:
        df = A.load_saved(name+".parquet")
        print(name, df.shape, list(df.columns)[:8])
        print(df.head(3))
        print()
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
