import numpy as np, pandas as pd

def fn(view, snapshot_day):
    P = lambda m: print(f'  [{snapshot_day}] {m}', flush=True)
    if snapshot_day != 207:
        tx0 = view.table('transactions')
        first = tx0.groupby('household_key')['day'].min()
        idx = pd.Index(first[first <= snapshot_day - 84].index)
        return pd.DataFrame({'cs_probe': 1.0}, index=idx)
    tx = view.table('transactions')
    tx = tx[['household_key','basket_id','day','sales_value']].copy()
    P(f'tx {tx.shape} dtypes {dict(tx.dtypes.astype(str))}')
    trips = tx[['household_key','basket_id','day']].drop_duplicates()
    P('trips')
    g = trips.sort_values(['household_key','day'])
    P('sorted')
    days = g['day'].to_numpy(float)
    gaps = days - g.groupby('household_key')['day'].shift().to_numpy()
    ok = ~np.isnan(gaps)
    med = pd.Series(gaps[ok]).groupby(g['household_key'].to_numpy()[ok]).median()
    P(f'med_gap n={len(med)}')
    w = tx.groupby(['household_key','wk' if 'wk' in tx else 'day'])['sales_value'].sum()
    P('groupby sum')
    # the exact op that died:
    tx['wk'] = (tx['day'] - 1) // 7
    w = tx.groupby(['household_key','wk'])['sales_value'].sum().unstack(fill_value=0.0)
    P(f'unstack {w.shape}')
    return pd.DataFrame({'cs_probe': w.sum(axis=1)})

feats = agent_api.build_features(fn)
print('OK:', feats.shape)
