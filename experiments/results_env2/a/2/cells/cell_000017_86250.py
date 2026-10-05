import agent_api as A, pandas as pd, numpy as np

def probe(view, snapshot_day):
    hh = view.households
    print('day', view.day, 'week', view.week, 'hh type', type(hh), 'n', 0 if hh is None else len(hh))
    if hh is not None: print(hh.head(3))
    tx = view.table('transactions')
    print('tx', tx.shape, 'maxday', tx.day.max())
    dm = view.table('display_mailer')
    print('dm', dm.shape)
    if snapshot_day==95:
        print(dm.display.value_counts().head()); print(dm.mailer.value_counts().head())
    pr = view.table('products')
    print('prod cols', pr.columns.tolist(), 'brand vals', pr.brand.unique()[:8])
    dem = view.table('demographics')
    print('dem', dem.shape)
    for c in ['classification_1','classification_3','classification_4','classification_5']:
        print(c, sorted(dem[c].unique())[:14])
    return hh.iloc[:2].to_frame('k') if hh is not None else pd.DataFrame()

out = A.build_features(probe)
print('out cols', out.columns.tolist(), out.shape)
