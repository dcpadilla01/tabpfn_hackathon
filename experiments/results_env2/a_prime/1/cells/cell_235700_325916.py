
import numpy as np, pandas as pd, time
t0=time.time()
snap = agent_api.snapshot()
tx = snap.transactions[['household_key','basket_id','day','product_id','sales_value','quantity']].copy()
tx = tx.merge(snap.products[['product_id','department']], on='product_id', how='left')
print("tx", tx.shape, "day range", tx.day.min(), tx.day.max())

hh_sorted = np.sort(tx.household_key.unique())
hh_i = {h:i for i,h in enumerate(hh_sorted)}
H = len(hh_i); DM = int(tx.day.max())
tx['hi'] = tx.household_key.map(hh_i)
days = np.arange(DM+1)

def cumpivot(col, how='sum'):
    if how=='sum':
        g = tx.groupby(['hi','day'])[col].sum()
    else:
        g = tx.groupby(['hi','day'])[col].nunique()
    P = g.unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0)
    P = P.reindex(range(H)).fillna(0.0).sort_index()
    return P.cumsum(axis=1).values

C = cumpivot('sales_value'); T = cumpivot('basket_id','nun'); Q = cumpivot('quantity')
Pd = tx.groupby(['hi','day'])['sales_value'].sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0).reindex(range(H)).fillna(0.0).sort_index()
E = Pd.ewm(halflife=28, axis=1).mean().values
wk = np.array([ (d+8)//7 for d in days ])
Wt = Pd.T.groupby(wk).sum().T
print("pivots done", round(time.time()-t0,1), "H",H, "weeks", Wt.shape[1])

ds = tx.groupby('hi')['day'].apply(lambda s: np.sort(s.values))
NEED = sorted(set(range(84,432,28)) | set(range(95,432,28)))
lastmat = np.zeros((H, len(NEED)), dtype=int)
for i in range(H):
    a = ds.iloc[i]
    p = np.searchsorted(a, NEED, side='right')-1
    lastmat[i] = a[p]
LAST = lastmat
print("lastday done", round(time.time()-t0,1))
print("NEED:", NEED)
