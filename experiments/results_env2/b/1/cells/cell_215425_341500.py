import numpy as np, pandas as pd
def fn(view, snapshot_day):
    print('day', snapshot_day, 'start', flush=True)
    a = np.zeros(50_000_000)  # 400MB
    print('  alloc 400MB ok', flush=True)
    b = np.zeros(190_000_000)  # ~1.5GB
    b[0] = 1.0
    print('  alloc 1.5GB ok', flush=True)
    del a, b
    tx = view.table('transactions')
    return pd.DataFrame({'cs_test': float(len(tx))}, index=pd.Index(['A'], name='household_key'))

feats = agent_api.build_features(fn)
print('memtest OK:', feats.shape)
