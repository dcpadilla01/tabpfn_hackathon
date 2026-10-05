import agent_api, pandas as pd, numpy as np
for name in ["candA_dm","candB_g","candC_hb","candD_su","candE_td","candGST"]:
    try:
        df = agent_api.load_saved(name+".parquet")
        print(name, df.shape)
        print(list(df.columns)[:40])
        print("---")
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
