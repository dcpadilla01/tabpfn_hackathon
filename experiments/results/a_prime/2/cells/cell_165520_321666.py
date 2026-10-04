import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
t['hh'] = t['household_key'].astype('category')
print(t.shape, t['hh'].nunique(), t['hh'].dtype)
p = agent_api.save_table(t, 'e015_hh_identity.parquet')
print(p)
