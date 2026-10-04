import agent_api as api
import pandas as pd, numpy as np

def fn(view, day):
    tx = view.transactions
    hh = view.households
    recent = tx[tx['day'] > day - 28]
    g = recent.groupby('household_key').agg(spend28=('sales_value','sum'), trips28=('basket_id','nunique'))
    last = tx.groupby('household_key')['day'].max()
    out = pd.DataFrame(index=pd.Index(hh, name='household_key'))
    out['spend28'] = g['spend28'].reindex(out.index).fillna(0.0)
    out['trips28'] = g['trips28'].reindex(out.index).fillna(0)
    out['recency'] = (day - last).reindex(out.index).fillna(999)
    return out

df = api.build_features(fn)
print(df.shape, df.columns.tolist())
print(df.head())
p = api.save_table(df, 'rfm28')
print(p)
