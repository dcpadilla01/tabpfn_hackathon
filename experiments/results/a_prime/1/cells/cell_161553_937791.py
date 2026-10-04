import agent_api as A
import pandas as pd

for name in ["e013_peers", "e014_demo_l13fix", "micro", "nf_compact19", "nf_seasonal"]:
    try:
        df = A.load_saved(name + ".parquet")
        print(name, df.shape)
        print(list(df.columns)[:40])
        print("---")
    except Exception as e:
        print(name, "ERR", e)