import pandas as pd, numpy as np

def probe(v, snap):
    tx = v.transactions
    d = {}
    d['v_day'] = v.day
    d['v_week'] = v.week
    d['tx_max_day'] = tx['day'].max() if len(tx) else -1
    d['tx_min_day'] = tx['day'].min() if len(tx) else -1
    d['tx_rows'] = len(tx)
    g = tx.groupby('household_key')['day'].agg(['min','max'])
    d['hh_min'] = g['min'].reindex(list(v.households))
    d['hh_max'] = g['max'].reindex(list(v.households))
    return pd.DataFrame(d, index=pd.Index(list(v.households), name='household_key'))

out = agent_api.build_features(probe)
print(out.groupby('snapshot_day')[['v_day','v_week','tx_max_day','tx_min_day','tx_rows']].first())
print(out.groupby('snapshot_day')[['hh_min','hh_max']].mean())
print(out['hh_min'].describe())
print('n hh with hh_min<=day-84:', (out['hh_min'] <= out['v_day']-84).mean())