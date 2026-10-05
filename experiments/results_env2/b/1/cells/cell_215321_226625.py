import pandas as pd, numpy as np
def fn(view, snapshot_day):
    tx = view.table('transactions')
    hh = view.households
    print('day', snapshot_day, 'tx rows', len(tx), 'hh type', type(hh).__name__, flush=True)
    if hh is None:
        first = tx.groupby('household_key')['day'].min()
        idx = pd.Index(first[first <= snapshot_day - 84].index)
    elif isinstance(hh, pd.DataFrame):
        idx = pd.Index(hh['household_key'].unique())
    else:
        idx = pd.Index(pd.unique(np.asarray(hh).ravel()))
    return pd.DataFrame({'cs_test': 1.0}, index=idx)

feats = agent_api.build_features(fn)
print('trivial OK:', feats.shape)
