
import numpy as np, pandas as pd

tt = agent_api.train_targets()
y = tt['future_spend_4w']
print("targets:", tt.shape, "zero share:", round(float((y==0).mean()),3))
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2))

e012 = agent_api.load_saved('e012_style.parquet')
print("\ne012:", e012.shape)
feats = [c for c in e012.columns if c not in ('household_key','snapshot_day')]
print("n_feats:", len(feats))
print("dtypes:", e012[feats].dtypes.value_counts().to_dict())
nonnum = [c for c in feats if not np.issubdtype(e012[c].dtype, np.number)]
print("non-numeric:", nonnum)
print("NaN cols>50%:", [c for c in feats if e012[c].isna().mean()>0.5][:10])

e015 = agent_api.load_saved('e015_stack.parquet')
print("\ne015 extra cols:", [c for c in e015.columns if c not in e012.columns])
print("snapshots:", sorted(e012.snapshot_day.unique()))
print("rows per snapshot:", e012.groupby('snapshot_day').size().to_dict())


# ---- cell ----

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


# ---- cell ----

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


# ---- cell ----

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
Wv = Wt.values  # columns = weeks 1..66 in order
ds = tx.groupby('hi')['day'].apply(lambda s: np.sort(s.values))
NEED = np.array(sorted(set(range(84,432,28)) | set(range(95,432,28))))
LAST = np.zeros((H, len(NEED)), dtype=int)
for i in range(H):
    a = ds.iloc[i]; LAST[i] = a[np.searchsorted(a, NEED, side='right')-1]
dept = tx.groupby(['hi','day','department'])['sales_value'].sum().reset_index()
depts = sorted(dept.department.unique())
Dcum = {}
for d in depts:
    g2 = dept[dept.department==d].groupby(['hi','day'])['sales_value'].sum().unstack(fill_value=0.0).reindex(columns=days, fill_value=0.0).reindex(range(H)).fillna(0.0).sort_index()
    Dcum[d] = g2.cumsum(axis=1).values
print("setup", round(time.time()-t0,1), "ndeps", len(depts))

rows = []
for gi, g in enumerate(NEED):
    last = LAST[:, gi]; ar = np.arange(H)
    w0, w1 = (g-27)//7 - 1, (g+1)//7   # positional cols: weeks g-27..g
    wk8 = Wv[:, w0-7:w1]
    wk8m = np.where(wk8.sum(axis=1, keepdims=True)>0, wk8, np.nan)
    wkm = np.nanmean(wk8m, axis=1); wks = np.nanstd(wk8m, axis=1)
    wkmx = np.nanmax(wk8m, axis=1); wkmn = np.nanmin(wk8m, axis=1)
    wkm2 = np.nanmean(wk8m[:, :4], axis=1)
    r = pd.DataFrame({'hi':ar, 'g':g,
        'sp28':cum_at(C,last)-cum_at(C,g-28) if False else C[ar,last]-C[ar,g-28],
        'sp56':C[ar,last]-C[ar,g-56], 'sp84':C[ar,last]-C[ar,g-84],
        'sp182':C[ar,last]-C[ar,g-182], 'sp365':C[ar,last]-C[ar,g-365],
        'tr28':T[ar,last]-T[ar,g-28], 'tr84':T[ar,last]-T[ar,g-84], 'tr182':T[ar,last]-T[ar,g-182],
        'q28':Q[ar,last]-Q[ar,g-28], 'ew':E[ar,last], 'wk28':Wv[:, w0:w1].sum(axis=1),
        'wkm':wkm, 'wks':wks, 'wkmx':wkmx, 'wkmn':wkmn, 'wkm2':wkm2})
    for d in depts:
        Dc = Dcum[d]; r['d_'+d] = Dc[ar,last]-Dc[ar,g-28]
    rows.append(r)
F = pd.concat(rows, ignore_index=True)
F['household_key'] = hh_sorted[F['hi'].values]; F.drop(columns='hi', inplace=True)
print("F", F.shape, round(time.time()-t0,1))
agent_api.save_table(F, 'rawrec')


# ---- cell ----

import pandas as pd, numpy as np
e012 = agent_api.load_saved('e012_style.parquet')
cols = [c for c in e012.columns if c not in ('household_key','snapshot_day')]
print(len(cols))
print(cols)
r = agent_api.load_saved('rawrec')
print(r.shape, r.columns.tolist()[:20])


# ---- cell ----

import pandas as pd, numpy as np
r = agent_api.load_saved('rawrec.parquet')
print(r.shape)
print(r.columns.tolist())
print(r.head(3))


# ---- cell ----

import pandas as pd, numpy as np, time
t0=time.time()
e012 = agent_api.load_saved('e012_style.parquet')
r = agent_api.load_saved('rawrec.parquet').rename(columns={'g':'snapshot_day'})
M = e012.merge(r, on=['household_key','snapshot_day'], how='left')
print(M.shape)
first = [c for c in r.columns if c not in ('household_key','snapshot_day')][0]
print("missing rows:", int(M[first].isna().sum()))
print("corr sp28 vs spend_28:", round(float(np.corrcoef(M.sp28, M.spend_28)[0,1]),4))
print("mean abs diff:", round(float((M.sp28-M.spend_28).abs().mean()),3))
print(M[['household_key','snapshot_day','sp28','spend_28','tr28','trips_28','ew','d_ewma_spend_hl28']].head(5))
print("secs", round(time.time()-t0,1))


# ---- cell ----

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


# ---- cell ----

import pandas as pd, numpy as np
e012 = agent_api.load_saved('e012_style.parquet')
r = agent_api.load_saved('rawrec.parquet').rename(columns={'g':'snapshot_day'})
M = e012.merge(r, on=['household_key','snapshot_day'], how='left')
first = 'sp28'
print("rows:", len(M), "missing:", int(M[first].isna().sum()))
print("corr sp28 vs spend_28:", round(float(np.corrcoef(M.sp28, M.spend_28)[0,1]),4))
print("mean abs diff sp28 vs spend_28:", round(float((M.sp28-M.spend_28).abs().mean()),3))
print(M[['household_key','snapshot_day','sp28','spend_28','tr28','trips_28']].head(6))
# check zero-target households: does sp28==0 align?
tt = agent_api.train_targets()
M2 = M.merge(tt, on=['household_key','snapshot_day'])
print("n train rows:", len(M2))
print("P(y=0 | sp28=0):", round(float((M2.loc[M2.sp28==0,'future_spend_4w']==0).mean()),3))
print("mean y | sp28=0:", round(float(M2.loc[M2.sp28==0,'future_spend_4w'].mean()),2))
print("mean y | sp28>0:", round(float(M2.loc[M2.sp28>0,'future_spend_4w'].mean()),2))


# ---- cell ----

import pandas as pd, numpy as np
e012 = agent_api.load_saved('e012_style.parquet')
r = agent_api.load_saved('rawrec.parquet').rename(columns={'g':'snapshot_day'})
print("e012 dtypes:", e012[['household_key','snapshot_day']].dtypes.to_dict())
print("rawrec dtypes:", r[['household_key','snapshot_day']].dtypes.to_dict())
print("e012 hh sample:", e012.household_key.head(3).tolist())
print("rawrec hh sample:", r.household_key.head(3).tolist())
print("e012 n unique hh:", e012.household_key.nunique(), "rawrec:", r.household_key.nunique())
m = set(zip(e012.household_key, e012.snapshot_day))
m2 = set(zip(r.household_key, r.snapshot_day))
print("e012 keys not in rawrec:", len(m-m2), "rawrec keys not in e012:", len(m2-m))
miss = list(m-m2)[:5]; print("examples:", miss)
# check those households in rawrec at other days
ex_hh = miss[0][0] if miss else None
if ex_hh is not None:
    print("rawrec rows for that hh:", r[r.household_key==ex_hh][['household_key','snapshot_day','sp28']].head(20))


# ---- cell ----

import pandas as pd, numpy as np
e012 = agent_api.load_saved('e012_style.parquet')
r = agent_api.load_saved('rawrec.parquet').rename(columns={'g':'snapshot_day'})
M = e012.merge(r, on=['household_key','snapshot_day'], how='left')
print("rows:", len(M), "missing:", int(M['sp28'].isna().sum()))
print("corr sp28 vs spend_28:", round(float(np.corrcoef(M.sp28, M.spend_28)[0,1]),4))
print("mean abs diff:", round(float((M.sp28-M.spend_28).abs().mean()),3))
# how many features total now
feats = [c for c in M.columns if c not in ('household_key','snapshot_day')]
print("total feature cols:", len(feats))
agent_api.save_table(M, 'e016_merged')
