
import agent_api, pandas as pd, numpy as np
rr = agent_api.load_saved('rawrec.parquet')
print('unique g:', sorted(rr['g'].unique()))
print('g+11 matches snapshot days?', set(np.array(sorted(rr['g'].unique()))+11) == set(agent_api.snapshot_days()['train']+agent_api.snapshot_days()['validation']))
# check for truncation artifact at late g: sp28 zeros share by g
print(rr.groupby('g')['sp28'].apply(lambda s: (s==0).mean()))
print(rr.groupby('g')['sp56'].mean())
