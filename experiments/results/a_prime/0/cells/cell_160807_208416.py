
import agent_api as A, pandas as pd, numpy as np
for name in ['hist_v1','hist_v2','structure_v1']:
    t = A.load_saved(name+'.parquet')
    print(name, t.shape)
    print(list(t.columns))
    print()
