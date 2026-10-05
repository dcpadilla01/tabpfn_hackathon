
import agent_api as A, pandas as pd
ewma = A.load_saved("ewma_block_v1.parquet")
mkt  = A.load_saved("mkt_v1.parquet")
comp = A.load_saved("comp_v1.parquet")
season = A.load_saved("season_demo_v1.parquet")
for nm, b in [("ewma",ewma),("mkt",mkt),("comp",comp)]:
    print(nm, b.shape)
    print(list(b.columns))
    print("---")
