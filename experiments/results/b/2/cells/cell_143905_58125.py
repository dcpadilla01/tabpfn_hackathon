import pandas as pd, numpy as np

def probe(v, snap):
    print('CALL snap=', snap, '| v.day =', v.day, '| v.week =', v.week,
          '| hh type', type(v.households), '| n', len(v.households))
    print('hh sample', list(v.households)[:3])
    return pd.DataFrame(index=pd.Index(list(v.households), name='household_key'))

b = agent_api.load_saved('e001_recent_spend.parquet')
print(b['snapshot_day'].value_counts().sort_index())
_ = agent_api.build_features(probe)
print('done')