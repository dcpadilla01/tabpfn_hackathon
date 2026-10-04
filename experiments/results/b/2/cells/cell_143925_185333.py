import pandas as pd, numpy as np

def probe(v, snap):
    print('CALL', snap, v.day, v.week, len(v.households), len(v.transactions))
    return pd.DataFrame({'x': 1.0}, index=pd.Index(list(v.households), name='household_key'))

out = agent_api.build_features(probe)
print('outer sees:', out.shape, list(out.columns))
print(out['snapshot_day'].value_counts().sort_index().head())
print(out.head())