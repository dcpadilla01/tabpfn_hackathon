import agent_api as A, pandas as pd, numpy as np

def fn(view, snapshot_day):
    sd = snapshot_day
    t = view.transactions
    p = view.products[['product_id','department','commodity_desc','manufacturer','brand']]
    m = t.merge(p, on='product_id', how='left')
    d = m['day'].values
    w = m[(d > sd-28) & (d <= sd)]
    idx = pd.Index(view.households, name='household_key')
    res = {
        'wcols': ' | '.join(w.columns),
        'wshape': str(w.shape),
        'gsum': str(w.groupby('household_key')['sales_value'].sum().sum()),
    }
    # also try filtering transactions first then merging
    t2 = t[(t.day > sd-28) & (t.day <= sd)]
    m2 = t2.merge(p, on='product_id', how='left')
    res['m2cols'] = ' | '.join(m2.columns)[:200]
    res['gsum2'] = str(m2.groupby('household_key')['sales_value'].sum().sum())
    return pd.DataFrame(res, index=[idx[0]])

f = A.build_features(fn)
for c in f.columns: print(c, '=>', f.iloc[0][c])