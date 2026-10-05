
import pandas as pd, numpy as np
e017 = agent_api.load_saved('e017_xsec_rank.parquet')
e018 = agent_api.load_saved('e018_tree_feats.parquet')
cols17 = set(e017.columns) - {'household_key','snapshot_day'}
cols18 = set(e018.columns) - {'household_key','snapshot_day'}
missing = sorted(cols18 - cols17)
print('E018 feats missing from E017:', missing)
# check near-duplicates by name similarity
import re
for m in missing:
    stem = re.sub(r'(w\d+|28|84|364)$', '', m)
    sim = [c for c in cols17 if stem and stem in c]
    print(m, '-> similar in E017:', sim[:8])
