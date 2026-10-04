import agent_api as A, pandas as pd, numpy as np

def fn(view, snapshot_day):
    t = view.transactions
    p = view.products[['product_id','department','commodity_desc','manufacturer','brand']]
    m = t.merge(p, on='product_id', how='left')
    idx = pd.Index(view.households, name='household_key')
    return pd.DataFrame({
        'tcols': ' | '.join(t.columns),
        'mcols': ' | '.join(m.columns),
        'n_tx': len(t), 'n_m': len(m),
    }, index=[idx[0]])

f = A.build_features(fn)
print(f.iloc[0,0]); print(f.iloc[0,1]); print(f.iloc[0,2], f.iloc[0,3])