import agent_api, pandas as pd, numpy as np
# Dry-run build_features to see what a view exposes inside fn
def probe(view, s):
    print('snapshot_day', s, 'day', view.day, 'week', view.week)
    print('households type:', type(view.households))
    hh = view.households
    try:
        print('len hh:', len(hh), 'first:', list(hh)[:3])
    except Exception as e:
        print('hh err', e)
    tx = view.transactions
    print('tx shape', tx.shape, 'max day', tx['day'].max(), 'min day', tx['day'].min())
    print('neg sales:', (tx.sales_value<0).sum(), 'zero sales:', (tx.sales_value==0).sum())
    print('cols:', tx.columns.tolist())
    return pd.DataFrame(index=hh if hh is not None else tx.household_key.unique())

out = agent_api.build_features(probe)
print('out shape:', out.shape)
print(out.head())
