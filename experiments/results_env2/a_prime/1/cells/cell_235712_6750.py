
import numpy as np, pandas as pd, time
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
wk = np.array([(d+8)//7 for d in days]); Wt = Pd.T.groupby(wk).sum().T
ds = tx.groupby('hi')['day'].apply(lambda s: np.sort(s.values))
NEED = np.array(sorted(set(range(84,432,28)) | set(range(95,432,28))))
LAST = np.zeros((H, len(NEED)), dtype=int)
for i in range(H):
    a = ds.iloc[i]; LAST[i] = a[np.searchsorted(a, NEED, side='right')-1]
print("setup", round(time.time()-t0,1))

# department daily spend pivot
dept = tx.groupby(['hi','day','department'])['sales_value'].sum().reset_index()
depts = sorted(dept.department.unique())
print("n depts:", len(depts))
Dcum = {}
for d in depts:
    g = dept[dept.department==d].groupby(['hi','day'])['sales_value'].sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0).reindex(range(H)).fillna(0.0).sort_index()
    Dcum[d] = g.cumsum(axis=1).values
print("dept cum done", round(time.time()-t0,1))

def cum_at(Cm, lastday):  # Cm hh x (DM+1); lastday array hh
    return Cm[np.arange(H), lastday]

rows = []
for gi, g in enumerate(NEED):
    last = LAST[:, gi]
    sp28 = cum_at(C, last)-cum_at(C, g-28); sp56 = cum_at(C, last)-cum_at(C, g-56)
    sp84 = cum_at(C, last)-cum_at(C, g-84); sp182 = cum_at(C, last)-cum_at(C, g-182)
    sp365 = cum_at(C, last)-cum_at(C, g-365)
    tr28 = cum_at(T, last)-cum_at(T, g-28); tr84 = cum_at(T, last)-cum_at(T, g-84)
    tr182 = cum_at(T, last)-cum_at(T, g-182)
    q28 = cum_at(Q, last)-cum_at(Q, g-28)
    ew = E[np.arange(H), last]
    wk28 = Wt[:, (g-27)//7:(g+1)//7].sum(axis=1)
    wk8 = Wt[:, (g-55)//7:(g+1)//7]
    wk8m = np.where(wk8.sum(axis=1, keepdims=True)>0, wk8, np.nan)
    wkm = np.nanmean(wk8m, axis=1); wks = np.nanstd(wk8m, axis=1)
    wkmx = np.nanmax(wk8m, axis=1); wkmn = np.nanmin(wk8m, axis=1)
    wkm2 = np.nanmean(wk8m[:, :4], axis=1)
    # dept 28d spend
    d28 = {d: cum_at(Dcum[d], last)-cum_at(Dcum[d], g-28) for d in depts}
    r = pd.DataFrame({'hi':np.arange(H), 'g':g, 'sp28':sp28, 'sp56':sp56, 'sp84':sp84,
        'sp182':sp182, 'sp365':sp365, 'tr28':tr28, 'tr84':tr84, 'tr182':tr182, 'q28':q28, 'ew':ew,
        'wk28':wk28, 'wkm':wkm, 'wks':wks, 'wkmx':wkmx, 'wkmn':wkmn, 'wkm2':wkm2})
    for d in depts: r['d_'+d] = d28[d]
    rows.append(r)
F = pd.concat(rows, ignore_index=True)
print("F", F.shape, round(time.time()-t0,1))
# map hi back to household_key
F['household_key'] = hh_sorted[F['hi'].values]
F.drop(columns='hi', inplace=True)
agent_api.save_table(F, 'rawrec')
