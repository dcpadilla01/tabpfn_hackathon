import agent_api, pandas as pd
for name in ["candC_hb","candD_su"]:
    df = agent_api.load_saved(name+".parquet")
    cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
    print(name, len(cols))
    print(cols[120:])
    print("===")
