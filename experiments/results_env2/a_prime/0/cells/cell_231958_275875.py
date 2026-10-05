import agent_api, pandas as pd, numpy as np
b = agent_api.load_saved("e012_outcome2.parquet")
print("shape:", b.shape)
cols = list(b.columns)
# group by prefix
import collections
groups = collections.OrderedDict()
for c in cols:
    p = c.split('_')[0] if '_' in c else c
    groups.setdefault(p, []).append(c)
for p, cs in groups.items():
    print(f"{p}: {len(cs)}", cs[:8], "..." if len(cs)>8 else "")
