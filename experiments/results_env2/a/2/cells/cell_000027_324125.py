import agent_api as A, pandas as pd, numpy as np

def probe(view, snapshot_day):
    hh = view.households
    print('day', view.day, 'week', view.week, 'n_hh', len(hh))
    tx = view.table('transactions')
    print('tx', tx.shape, 'maxday', int(tx.day.max()))
    dm = view.table('display_mailer')
    print('dm', dm.shape, 'weeks', dm.week_no.min(), dm.week_no.max())
    if snapshot_day==95:
        print(dm.display.value_counts()); print(dm.mailer.value_counts())
    pr = view.table('products')
    print('prod cols', pr.columns.tolist(), 'brand vals', list(pr.brand.unique()[:8]))
    dem = view.table('demographics')
    print('dem', dem.shape)
    for c in ['classification_1','classification_3','classification_4','classification_5']:
        print(c, sorted(dem[c].unique())[:14])
    return pd.DataFrame(index=hh[:2])

out = A.build_features(probe)
print('out', out.shape)
