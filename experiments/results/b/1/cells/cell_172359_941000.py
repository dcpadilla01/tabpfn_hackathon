import agent_api as A
import pandas as pd, numpy as np
tgt = A.load_saved('my_targets.parquet')
cand = A.load_saved('e019_cand.parquet')
print(cand.dtypes.value_counts())
print([c for c in cand.columns if cand[c].dtype.kind not in 'ifb'][:20])