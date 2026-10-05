import agent_api as A
import pandas as pd, numpy as np

def fn(view, snapshot_day):
    h = view.households
    print('type:', type(h))
    try:
        print('len:', len(h))
    except Exception as e:
        print('len err', e)
    print(repr(h)[:200])
    t = view.table('transactions')
    print('tx shape', t.shape, 'max day', t.day.max())
    print('view.day', view.day, 'view.week', view.week)
    return pd.DataFrame({'x': 1.0}, index=list(h)[:5] if not isinstance(h, pd.Index) else h[:5])

res = A.build_features(fn)
print(res.shape)
print(res.head())
