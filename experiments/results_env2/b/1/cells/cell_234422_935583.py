import agent_api, pandas as pd, numpy as np
for name in ["candA_dm","candB_g","candC_hb","candD_su","candE_td","candGST"]:
    df = agent_api.load_saved(name+".parquet")
    cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    print(name, len(cols))
    print(cols[60:])
    print("===")
