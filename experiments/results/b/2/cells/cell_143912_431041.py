import pandas as pd, numpy as np

info = []
def probe(v, snap):
    info.append({'snap': snap, 'day': v.day, 'week': v.week,
                 'nhh': len(v.households), 'hh0': list(v.households)[0],
                 'ntx': len(v.transactions)})
    return pd.DataFrame(index=pd.Index(list(v.households), name='household_key'))

_ = agent_api.build_features(probe)
print(pd.DataFrame(info))