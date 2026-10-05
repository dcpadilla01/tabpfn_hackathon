
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
t0=time.time()
snap = agent_api.snapshot()
tx = snap.transactions[['household_key','basket_id','day','product_id','sales_value','quantity']].copy()
tx = tx.merge(snap.products[['product_id','department']], on='product_id', how='left')
hh_sorted = np.sort(tx.household_key.unique()); hh_i = {h:i for i,h in enumerate(hh_sorted)}
H = len(hh_i); DM = int(tx.day.max()); tx['hi'] = tx.household_key.map(hh_i)
days = np.arange(DM+1)
def cumpivot(col, how='sum'):
    g = tx.groupby(['hi','day'])[col].sum() if how=='sum' else tx.groupby(['hi','day'])[col].nunique()
    P = g.unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0).reindex(range(H)).fillna(0.0).sort_index()
    return P.cumsum(axis=1).values
C = cumpivot('sales_value'); T = cumpivot('basket_id','nun'); Q = cumpivot('quantity')
Pd = tx.groupby(['hi','day'])['sales_value'].sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0).reindex(range(H)).fillna(0.0).sort_index()
E = Pd.ewm(halflife=28, axis=1).mean().values
wk = np.array([(d+8)//7 for d in days]); Wt = Pd.T.groupby(wk).sum().T; Wv = Wt.values
ds = tx.groupby('hi')['day'].apply(lambda s: np.sort(s.values))
NEED = np.array(sorted(set(range(84,460,28)) | set(range(95,460,28))))
LAST = np.zeros((H, len(NEED)), dtype=int)
for i in range(H):
    a = ds.iloc[i]; LAST[i] = a[np.searchsorted(a, NEED, side='right')-1]
dept = tx.groupby(['hi','day','department'])['sales_value'].sum().reset_index()
depts = sorted(dept.department.unique())
Dcum = {}
for d in depts:
    g2 = dept[dept.department==d].groupby(['hi','day'])['sales_value'].sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0).reindex(range(H)).fillna(0.0).sort_index()
    Dcum[d] = g2.cumsum(axis=1).values
rows = []
for gi, g in enumerate(NEED):
    last = LAST[:, gi]; ar = np.arange(H)
    w0, w1 = (g-27)//7 - 1, (g+1)//7
    wk8 = Wv[:, w0-7:w1]
    wk8m = np.where(wk8.sum(axis=1, keepdims=True)>0, wk8, np.nan)
    r = pd.DataFrame({'hi':ar, 'g':g,
        'sp28':C[ar,last]-C[ar,g-28], 'sp56':C[ar,last]-C[ar,g-56], 'sp84':C[ar,last]-C[ar,g-84],
        'sp182':C[ar,last]-C[ar,g-182], 'sp365':C[ar,last]-C[ar,g-365],
        'tr28':T[ar,last]-T[ar,g-28], 'tr84':T[ar,last]-T[ar,g-84], 'tr182':T[ar,last]-T[ar,g-182],
        'q28':Q[ar,last]-Q[ar,g-28], 'ew':E[ar,last], 'wk28':Wv[:, w0:w1].sum(axis=1),
        'wkm':np.nanmean(wk8m,axis=1), 'wks':np.nanstd(wk8m,axis=1),
        'wkmx':np.nanmax(wk8m,axis=1), 'wkmn':np.nanmin(wk8m,axis=1), 'wkm2':np.nanmean(wk8m[:,:4],axis=1)})
    for d in depts:
        Dc = Dcum[d]; r['d_'+d] = Dc[ar,last]-Dc[ar,g-28]
    rows.append(r)
F = pd.concat(rows, ignore_index=True)
F['household_key'] = hh_sorted[F['hi'].values]; F.drop(columns='hi', inplace=True)
print("F", F.shape, "g values:", sorted(F.g.unique())[:5], "...", sorted(F.g.unique())[-5:])
agent_api.save_table(F, 'rawrec')
print("secs", round(time.time()-t0,1))
