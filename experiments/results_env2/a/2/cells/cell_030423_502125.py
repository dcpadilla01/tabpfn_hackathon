import agent_api as A, pandas as pd, numpy as np
try:
    oof = A.load_saved("oof_e016_cv.parquet")
    print("ok", oof.shape, oof.columns.tolist())
    print(oof.head(3).to_string())
except Exception as e:
    print("ERR", type(e).__name__, str(e)[:200])
