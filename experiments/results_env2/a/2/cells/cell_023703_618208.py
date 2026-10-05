
import agent_api as A
import pandas as pd

for name in ["oof_e008","oof_e016_cv","oof_harness","oof_pt","pred_e008","pred_e009","pred_e014","pred_e016","pred_log1p","pred_pt","pred_seasonal","past_targets","feats_v3","feats_seasonal"]:
    try:
        df = A.load_saved(name+".parquet")
        print("==", name, df.shape)
        print(df.head(3))
        print("cols:", list(df.columns)[:20])
    except Exception as e:
        print("==", name, "ERR", type(e).__name__, e)
