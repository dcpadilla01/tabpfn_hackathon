
import agent_api, pandas as pd, numpy as np
rr = agent_api.load_saved('rawrec.parquet')
print(rr.dtypes)
print(rr.head(10))
print('nunique hh', rr['household_key'].nunique() if 'household_key' in rr.columns else 'no hh col')
