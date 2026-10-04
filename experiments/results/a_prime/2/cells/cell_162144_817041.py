import agent_api, pandas as pd, numpy as np
def probe(view, s):
    print('day', view.day, 'week', view.week)
    tx = view.transactions
    print('tx', tx.shape, 'days', tx.day.min(), tx.day.max())
    hh = np.asarray(view.households).ravel().tolist() if view.households is not None else tx.household_key.unique()
    print('n hh', len(hh), 'type', type(view.households))
    # first purchase day per household
    fd = tx.groupby('household_key')['day'].min()
    print('hh with first purchase <=', s-84, ':', (fd<=s-84).sum(), 'of', len(fd))
    print('example hh days:', fd.head())
    return pd.DataFrame(index=hh)

out = agent_api.build_features(probe)
print(out.shape)
