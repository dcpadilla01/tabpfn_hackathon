import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
# Build candidate features inside build_features: market seasonal level, holiday flags, price level, discount mix, dept mix extra
def fn(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[tx.sales_value.notna()]
    week = (snapshot_day+8)//7
    tx = tx.assign(w=(tx.day+8)//7)
    out = pd.DataFrame(index=view.households.index if view.households is not None else pd.Index([]))
    return out
# first check what view.households is inside build_features at a train snapshot
def fn2(view, snapshot_day):
    print('inside: day', view.day, 'week', view.week, 'households', type(view.households))
    tx = view.table('transactions')
    print('tx max day', tx.day.max(), 'shape', tx.shape)
    return pd.DataFrame(index=pd.Index([], name='household_key'))
import agent_api
bf = agent_api.build_features(fn2)
print(bf.shape)
