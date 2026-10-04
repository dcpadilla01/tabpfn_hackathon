import agent_api as A, pandas as pd, numpy as np

def fn(view, snapshot_day):
    t = view.transactions
    info = [list(t.columns), type(t)]
    p = view.products[['product_id','department','commodity_desc','manufacturer','brand']]
    m = t.merge(p, on='product_id', how='left')
    info.append(list(m.columns))
    print(info)
    return pd.DataFrame(index=pd.Index(view.households, name='household_key'))

f = A.build_features(fn)
print(f.shape)