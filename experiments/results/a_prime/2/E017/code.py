import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
print(t.shape, t.columns[:5].tolist(), t.columns[-5:].tolist())
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)
key = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in key]
num = df[feats].select_dtypes(include=[np.number]).columns.tolist()
nonnum = [c for c in feats if c not in num]
print('n feats', len(feats), 'numeric', len(num), 'nonnum', nonnum)
print('zero-target share', (df.future_spend_4w==0).mean())
tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
print('tr', len(tr), 'iv', len(iv))

def ev(cols, mode='raw', alpha=10.0, topk=None):
    A = tr[cols].astype(float).copy(); B = iv[cols].astype(float).copy()
    if mode=='log1p':
        for c in cols:
            if A[c].min() >= 0 and A[c].max() > 10:
                A[c] = np.log1p(A[c]); B[c] = np.log1p(B[c])
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    if topk:
        w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
        idx=np.argsort(-np.abs(w))[:topk]
        return ev([cols[i] for i in idx], mode, alpha)
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()

for a in [1.0,10.0,100.0,300.0]:
    r=ev(num,alpha=a); print('alpha %g: train %.3f inner-val %.3f'%(a,r[0],r[1]))
print('log1p a10:', ev(num,mode='log1p',alpha=10.0))
print('top100 a10:', ev(num,alpha=10.0,topk=100))
print('top200 a10:', ev(num,alpha=10.0,topk=200))
cm = df[num].corrwith(df.future_spend_4w)
print(cm.abs().sort_values(ascending=False).head(25))
print(df.groupby('snapshot_day').future_spend_4w.agg(['mean','count']))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, re

v = agent_api.snapshot(459)
tx = v.transactions.merge(v.products[['product_id','department']], on='product_id', how='left')
tx['department'] = tx['department'].fillna('UNK')
ds = tx.groupby('department')['sales_value'].sum().sort_values(ascending=False)
top = list(ds.head(24).index)
print('top depts', top)
print('coverage %.3f' % (ds.head(24).sum()/ds.sum()))

def sanitize(s): return re.sub(r'[^A-Za-z0-9]+','_',str(s))[:30]

def fn(view, snapshot_day):
    s = snapshot_day
    keys = np.asarray(view.households)
    tx = view.transactions.merge(view.products[['product_id','department']], on='product_id', how='left')
    tx['department'] = tx['department'].fillna('UNK')
    w = tx[(tx.day > s-364) & (tx.day <= s-336)]   # year-ago aligned 4-week window
    res = pd.DataFrame(index=keys)
    if len(w):
        piv = w.pivot_table(index='household_key', columns='department', values='sales_value', aggfunc='sum')
        piv = piv.reindex(columns=top).fillna(0.0).reindex(keys).fillna(0.0)
        tot = w.groupby('household_key')['sales_value'].sum().reindex(keys).fillna(0.0).values
    else:
        piv = pd.DataFrame(0.0, index=keys, columns=top)
        tot = np.zeros(len(keys))
    for d in top:
        res['dt13_'+sanitize(d)] = piv[d].values
    for d in top:
        res['ds13_'+sanitize(d)] = piv[d].values/(tot+1.0)
    return res

new = agent_api.build_features(fn)
print('new', new.shape)
base = agent_api.load_saved('e009_ewma_longlags.parquet')
m = base.merge(new, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
tt = agent_api.train_targets()
df = m.merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
def ev(cols, alpha=10.0):
    A = tr[cols].astype(float); B = iv[cols].astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()
bcols=[c for c in base.columns if c not in ('household_key','snapshot_day')]
acols=[c for c in m.columns if c not in ('household_key','snapshot_day')]
print('proxy base:', ev(bcols))
print('proxy +dept-season:', ev(acols))
p = agent_api.save_table(m, 'e017_dept_season.parquet')
print('saved', p)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, re

v = agent_api.snapshot(459)
tx = v.transactions.merge(v.products[['product_id','department']], on='product_id', how='left')
tx['department'] = tx['department'].astype(str).fillna('UNK')
ds = tx.groupby('department')['sales_value'].sum().sort_values(ascending=False)
top = list(ds.head(24).index)
print('top depts', top[:8], 'coverage %.3f' % (ds.head(24).sum()/ds.sum()))

def sanitize(s): return re.sub(r'[^A-Za-z0-9]+','_',str(s))[:30]

def fn(view, snapshot_day):
    s = snapshot_day
    keys = np.asarray(view.households)
    tx = view.transactions.merge(view.products[['product_id','department']], on='product_id', how='left')
    tx['department'] = tx['department'].astype(str)
    tx.loc[tx.department.isin(['nan','None','']), 'department'] = 'UNK'
    w = tx[(tx.day > s-364) & (tx.day <= s-336)]
    res = pd.DataFrame(index=keys)
    if len(w):
        piv = w.pivot_table(index='household_key', columns='department', values='sales_value', aggfunc='sum')
        piv = piv.reindex(columns=top).fillna(0.0).reindex(keys).fillna(0.0)
        tot = w.groupby('household_key')['sales_value'].sum().reindex(keys).fillna(0.0).values
    else:
        piv = pd.DataFrame(0.0, index=keys, columns=top)
        tot = np.zeros(len(keys))
    for d in top:
        res['dt13_'+sanitize(d)] = piv[d].values
    for d in top:
        res['ds13_'+sanitize(d)] = piv[d].values/(tot+1.0)
    return res

new = agent_api.build_features(fn)
print('new', new.shape)
base = agent_api.load_saved('e009_ewma_longlags.parquet')
m = base.merge(new, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
tt = agent_api.train_targets()
df = m.merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
def ev(cols, alpha=100.0):
    A = tr[cols].astype(float); B = iv[cols].astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()
bcols=[c for c in base.columns if c not in ('household_key','snapshot_day')]
acols=[c for c in m.columns if c not in ('household_key','snapshot_day')]
print('proxy base:', ev(bcols))
print('proxy +dept-season:', ev(acols))
p = agent_api.save_table(m, 'e017_dept_season.parquet')
print('saved', p)

# ---- cell ----
import agent_api, pandas as pd, numpy as np, re

v = agent_api.snapshot(459)
tx = v.transactions.merge(v.products[['product_id','department']], on='product_id', how='left')
tx['department'] = tx['department'].astype(str)
tx.loc[tx.department.isin(['nan','None','']), 'department'] = 'UNK'
ds = tx.groupby('department')['sales_value'].sum().sort_values(ascending=False)
top = list(ds.head(24).index)

def sanitize(s): return re.sub(r'[^A-Za-z0-9]+','_',str(s))[:30]

def fn(view, snapshot_day):
    s = snapshot_day
    keys = np.asarray(view.households)
    tx = view.transactions.merge(view.products[['product_id','department']], on='product_id', how='left')
    tx['department'] = tx['department'].astype(str)
    tx.loc[tx.department.isin(['nan','None','']), 'department'] = 'UNK'
    w = tx[(tx.day > s-364) & (tx.day <= s-336)]
    res = pd.DataFrame(index=keys)
    if len(w):
        piv = w.pivot_table(index='household_key', columns='department', values='sales_value', aggfunc='sum')
        piv = piv.reindex(columns=top).fillna(0.0).reindex(keys).fillna(0.0)
        tot = w.groupby('household_key')['sales_value'].sum().reindex(keys).fillna(0.0).values
    else:
        piv = pd.DataFrame(0.0, index=keys, columns=top)
        tot = np.zeros(len(keys))
    for d in top:
        res['dt13_'+sanitize(d)] = piv[d].values
    for d in top:
        res['ds13_'+sanitize(d)] = piv[d].values/(tot+1.0)
    return res

new = agent_api.build_features(fn)
base = agent_api.load_saved('e009_ewma_longlags.parquet')
m = base.merge(new, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
tt = agent_api.train_targets()
df = m.merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
def ev(cols, alpha=100.0):
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()
bcols=[c for c in base.columns if c not in ('household_key','snapshot_day')]
acols=[c for c in m.columns if c not in ('household_key','snapshot_day')]
print('proxy base:', ev(bcols))
print('proxy +dept-season:', ev(acols))
newcols=[c for c in m.columns if c.startswith(('dt13_','ds13_'))]
cm = df[newcols].corrwith(df.future_spend_4w).abs().sort_values(ascending=False)
print(cm.head(10))
p = agent_api.save_table(m, 'e017_dept_season.parquet')
print('saved', p)

# ---- cell ----
import agent_api, pandas as pd, numpy as np

tt = agent_api.train_targets()
def load(name):
    t = agent_api.load_saved(name)
    return t.merge(tt, on=['household_key','snapshot_day'], how='inner')

def ev(df, alpha=100.0):
    tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
    cols=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()

base = load('e009_ewma_longlags.parquet')
b = ev(base); print('E009  proxy: train %.2f inner %.2f  (harness 62.292)'%b)
known = {'e011_pruned.parquet':('E011',62.399), 'e013_full.parquet':('E013',62.518),
         'e014_full_pool.parquet':('E014',62.766), 'e016_dec_predictors.parquet':('E016',62.680)}
for nm,(hid,hm) in known.items():
    r = ev(load(nm))
    print('%s proxy: train %.2f inner %.2f  d_inner %+.2f | harness d %+.3f'%(hid, r[0], r[1], r[1]-b[1], hm-62.292))

# ---- cell ----
import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
cols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
print('n cols', len(cols))
import collections
pref = collections.Counter(c.split('_')[0] for c in cols)
print(dict(pref))

# check year-ago-style columns for train/val distribution mismatch
sus = [c for c in cols if 't13' in c or c.startswith('tlag_13') or 'lag_13' in c]
print('suspects:', sus)
for c in sus[:6]:
    g = df.groupby('snapshot_day')[c].agg(['mean','std'])
    print(c); print(g.tail(6).round(2)); print(g.head(3).round(2))

# fraction of rows with tlag_13==0 by period
for c in sus[:3]:
    z = (df[c]==0).groupby(df.snapshot_day<=347).mean()
    print(c,'zero-share train %.2f inner-val %.2f'% (z[True], z[False]))

# ---- cell ----
import agent_api, pandas as pd, numpy as np, re, collections

base = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()

# --- Variant 1: drop shift-prone year-ago columns ---
drop_pref = ('tlag_13','wyoy','yoy','index')
drop_cols = [c for c in base.columns if any(c==p or c.startswith(p+'_') or c.startswith(p) for p in drop_pref)]
print('dropping', len(drop_cols), drop_cols)
v1 = base.drop(columns=drop_cols)

# --- Variant 2: household seasonality + stride-7 zero-rate/hazard ---
def fn(view, snapshot_day):
    s = snapshot_day
    keys = np.asarray(view.households)
    tx = view.transactions
    res = pd.DataFrame(index=keys)
    wk = (tx.day + 8)//7
    tx = tx.assign(wk=wk, seas=wk % 52)
    nxt = [(s + 8)//7 + 1 + k for k in range(4)]  # weeks covering days s+1..s+28
    nxt_seas = [w % 52 for w in nxt]
    # household seasonal profile by week-of-year (mod 52), all history
    g = tx.groupby(['household_key','seas'])['sales_value'].sum()
    hh_tot = tx.groupby('household_key')['sales_value'].sum()
    prof = g.unstack('seas')
    prof = prof.reindex(keys).fillna(0.0)
    nweeks_obs = tx.groupby('household_key')['wk'].nunique().reindex(keys).fillna(1.0)
    weekly_mean = (hh_tot.reindex(keys).fillna(0.0) / nweeks_obs).replace(0, np.nan)
    pmean = prof.mean(axis=1)
    rel = (prof.div(pmean.replace(0,np.nan), axis=0))
    seas_score = np.nanmean(rel[nxt_seas].values, axis=1)
    res['hh_season_next4'] = pd.Series(np.nan_to_num(seas_score), index=keys)
    res['hh_season_next4_x'] = pd.Series(np.nan_to_num(seas_score), index=keys) * pd.Series(weekly_mean.values, index=keys).fillna(0).values
    # global seasonality: all-household spend by seas / mean
    gseas = tx.groupby('seas')['sales_value'].sum()
    gm = gseas.mean()
    res['glob_season_next4'] = float(np.mean([gseas.get(ss, 0.0)/gm for ss in nxt_seas]))
    # stride-7 aligned 28d windows: zero-rate and mean over last 12 windows
    zr = np.zeros(len(keys)); wm = np.zeros(len(keys)); wsd = np.zeros(len(keys))
    hh = pd.Series(tx.household_key.values, index=tx.index)
    dayv = tx.day.values; spv = tx.sales_value.values
    hhmap = {h:i for i,h in enumerate(keys)}
    wins = []
    for k in range(12):
        hi, lo = s - 7*k, s - 28 - 7*k
        if lo < 1: break
        wins.append((hi,lo))
    if wins:
        m = tx[((tx.day <= wins[0][0]) & (tx.day > wins[-1][1]))]
        m = m[m.household_key.isin(hhmap)]
        m = m.assign(hi=m.household_key.map(hhmap))
        arr = np.zeros((len(keys), len(wins)))
        for j,(hi,lo) in enumerate(wins):
            sub = m[(m.day <= hi) & (m.day > lo)]
            agg = sub.groupby('hi')['sales_value'].sum()
            arr[agg.index.values, j] = agg.values
        zr = (arr == 0).mean(axis=1)
        wm = arr.mean(axis=1); wsd = arr.std(axis=1)
    res['zrate_7s'] = zr
    res['wmean_7s'] = wm
    res['wcv_7s'] = np.where(wm>0, wsd/np.maximum(wm,1e-9), 0.0)
    # hazard: days since last / median inter-trip gap
    last = tx.groupby('household_key')['day'].max().reindex(keys)
    gaps = tx.sort_values('day').groupby('household_key')['day'].apply(lambda x: np.diff(np.unique(x.values)).mean() if len(np.unique(x))>1 else np.nan)
    med_gap = gaps.reindex(keys)
    res['due_ratio'] = (last.reindex(keys).fillna(s) - s).abs() / med_gap.replace(0,np.nan)
    res['due_ratio'] = res['due_ratio'].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return res

new = agent_api.build_features(fn)
print('new', new.shape, list(new.columns))
m2 = base.merge(new, on=['household_key','snapshot_day'], how='inner')
print('m2', m2.shape)

def ev(df, alpha=100.0):
    df = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
    cols=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()

b = ev(base); print('E009 base: %.2f / %.2f'%b)
r1 = ev(v1); print('v1 drop-shift: %.2f / %.2f  d_inner %+.2f'%(r1[0],r1[1],r1[1]-b[1]))
r2 = ev(m2); print('v2 +season/hazard: %.2f / %.2f  d_inner %+.2f'%(r2[0],r2[1],r2[1]-b[1]))
nc = [c for c in new.columns if c not in ('household_key','snapshot_day')]
df2 = m2.merge(tt, on=['household_key','snapshot_day'])
print(df2[nc].corrwith(df2.future_spend_4w))
print('shift check (train vs inner-val means):')
iv2 = df2[df2.snapshot_day>=375]; tr2 = df2[df2.snapshot_day<=347]
for c in nc:
    print('  %-18s tr %8.2f iv %8.2f' % (c, tr2[c].mean(), iv2[c].mean()))
p = agent_api.save_table(m2, 'e017_season_hazard.parquet'); print('saved', p)

# ---- cell ----
import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
drop_cols = [c for c in base.columns if c in ('tlag_13','index','wyoy','wyoy_ratio','yoy_spend364','yoy_ratio')]
v1 = base.drop(columns=drop_cols)

def fn(view, snapshot_day):
    s = snapshot_day
    keys = np.asarray(view.households)
    tx = view.transactions
    res = pd.DataFrame(index=keys)
    wk = (tx.day + 8)//7
    tx = tx.assign(wk=wk, seas=wk % 52)
    nxt = [(s + 8)//7 + 1 + k for k in range(4)]
    nxt_seas = [w % 52 for w in nxt]
    g = tx.groupby(['household_key','seas'])['sales_value'].sum()
    hh_tot = tx.groupby('household_key')['sales_value'].sum()
    prof = g.unstack('seas').reindex(columns=range(52))
    prof = prof.reindex(keys).fillna(0.0)
    nweeks_obs = tx.groupby('household_key')['wk'].nunique().reindex(keys).fillna(1.0)
    weekly_mean = (hh_tot.reindex(keys).fillna(0.0) / nweeks_obs).replace(0, np.nan)
    pmean = prof.mean(axis=1)
    rel = (prof.div(pmean.replace(0,np.nan), axis=0))
    seas_score = np.nanmean(rel[nxt_seas].values, axis=1)
    res['hh_season_next4'] = pd.Series(np.nan_to_num(seas_score), index=keys)
    res['hh_season_next4_x'] = pd.Series(np.nan_to_num(seas_score), index=keys) * pd.Series(weekly_mean.values, index=keys).fillna(0).values
    gseas = tx.groupby('seas')['sales_value'].sum()
    gm = gseas.mean()
    res['glob_season_next4'] = float(np.mean([gseas.get(ss, 0.0)/gm for ss in nxt_seas]))
    zr = np.zeros(len(keys)); wm = np.zeros(len(keys)); wsd = np.zeros(len(keys))
    hhmap = {h:i for i,h in enumerate(keys)}
    wins = []
    for k in range(12):
        hi, lo = s - 7*k, s - 28 - 7*k
        if lo < 1: break
        wins.append((hi,lo))
    if wins:
        m = tx[((tx.day <= wins[0][0]) & (tx.day > wins[-1][1]))]
        m = m[m.household_key.isin(hhmap)]
        m = m.assign(hi=m.household_key.map(hhmap))
        arr = np.zeros((len(keys), len(wins)))
        for j,(hi,lo) in enumerate(wins):
            sub = m[(m.day <= hi) & (m.day > lo)]
            agg = sub.groupby('hi')['sales_value'].sum()
            arr[agg.index.values, j] = agg.values
        zr = (arr == 0).mean(axis=1)
        wm = arr.mean(axis=1); wsd = arr.std(axis=1)
    res['zrate_7s'] = zr
    res['wmean_7s'] = wm
    res['wcv_7s'] = np.where(wm>0, wsd/np.maximum(wm,1e-9), 0.0)
    last = tx.groupby('household_key')['day'].max().reindex(keys)
    gaps = tx.sort_values('day').groupby('household_key')['day'].apply(lambda x: np.diff(np.unique(x.values)).mean() if len(np.unique(x))>1 else np.nan)
    med_gap = gaps.reindex(keys)
    res['due_ratio'] = ((last.reindex(keys).fillna(s) - s).abs() / med_gap.replace(0,np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return res

new = agent_api.build_features(fn)
print('new', new.shape, list(new.columns))
m2 = base.merge(new, on=['household_key','snapshot_day'], how='inner')
print('m2', m2.shape)

def ev(df, alpha=100.0):
    df = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
    cols=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()

b = ev(base); print('E009 base: %.2f / %.2f'%b)
r1 = ev(v1); print('v1 drop-shift: %.2f / %.2f  d_inner %+.2f'%(r1[0],r1[1],r1[1]-b[1]))
r2 = ev(m2); print('v2 +season/hazard: %.2f / %.2f  d_inner %+.2f'%(r2[0],r2[1],r2[1]-b[1]))
nc = [c for c in new.columns if c not in ('household_key','snapshot_day')]
df2 = m2.merge(tt, on=['household_key','snapshot_day'])
print(df2[nc].corrwith(df2.future_spend_4w))
iv2 = df2[df2.snapshot_day>=375]; tr2 = df2[df2.snapshot_day<=347]
for c in nc:
    print('  %-18s tr %8.2f iv %8.2f' % (c, tr2[c].mean(), iv2[c].mean()))
p = agent_api.save_table(m2, 'e017_season_hazard.parquet'); print('saved', p)

# ---- cell ----
import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e009_ewma_longlags.parquet')
m2 = agent_api.load_saved('e017_season_hazard.parquet')
tt = agent_api.train_targets()

def fn(view, snapshot_day):
    s = snapshot_day
    keys = np.asarray(view.households)
    tx = view.transactions
    res = pd.DataFrame(index=keys)
    hhmap = {h:i for i,h in enumerate(keys)}
    # stride-7 28d windows, last 16
    wins = [(s-7*k, s-28-7*k) for k in range(16) if s-28-7*k >= 1]
    arr = np.zeros((len(keys), len(wins)))
    if wins:
        m = tx[tx.household_key.isin(hhmap)].copy()
        m['hi'] = m.household_key.map(hhmap)
        for j,(hi,lo) in enumerate(wins):
            sub = m[(m.day <= hi) & (m.day > lo)]
            agg = sub.groupby('hi')['sales_value'].sum()
            arr[agg.index.values, j] = agg.values
    zr = (arr==0).mean(axis=1)
    wm = arr.mean(axis=1); wsd = arr.std(axis=1)
    res['zrate_7s'] = zr
    res['wmean_7s'] = wm
    res['wcv_7s'] = np.where(wm>0, wsd/np.maximum(wm,1e-9), 0.0)
    res['wmax_7s'] = arr.max(axis=1)
    res['zrate_recent'] = (arr[:,:6]==0).mean(axis=1)
    res['zrate_older'] = (arr[:,6:12]==0).mean(axis=1)
    res['zdiff'] = res['zrate_older'] - res['zrate_recent']
    # inter-trip gap stats (last 182d)
    t2 = tx[(tx.day > s-182) & (tx.day <= s)]
    g = t2.sort_values('day').groupby('household_key')['day'].apply(
        lambda x: np.diff(np.unique(x.values)) if len(np.unique(x))>1 else np.array([np.nan]))
    gm = g.apply(lambda a: np.nanmean(a)).reindex(keys).astype(float)
    gs = g.apply(lambda a: np.nanstd(a)).reindex(keys).astype(float)
    res['gap_mean'] = gm.fillna(84.0)
    res['gap_std'] = gs.fillna(0.0)
    res['gap_cv'] = (res['gap_std']/res['gap_mean'].replace(0,np.nan)).fillna(0.0)
    last = tx.groupby('household_key')['day'].max().reindex(keys).fillna(s)
    res['due_ratio'] = ((last - s).abs()/res['gap_mean'].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    res['p_active_x_wmean'] = (1-zr)*wm
    return res

new = agent_api.build_features(fn)
print('new', new.shape, [c for c in new.columns if c not in ('household_key','snapshot_day')])
m3 = base.merge(new, on=['household_key','snapshot_day'], how='inner')

def ev(df, alpha=100.0):
    df = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
    cols=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()

b = ev(base); print('E009 base: %.2f / %.2f'%b)
r2 = ev(m2); print('v2 season/hazard: %.2f / %.2f  d %+.2f'%(r2[0],r2[1],r2[1]-b[1]))
r3 = ev(m3); print('v3 extended:      %.2f / %.2f  d %+.2f'%(r3[0],r3[1],r3[1]-b[1]))
# v4: v3 + interactions with ewma/spend
m4 = m3.copy()
m4['ewma4_x_pact'] = m4['ewma_4']*(1-m4['zrate_7s'])
m4['spend28_x_wcv'] = m4['spend_28']*m4['wcv_7s']
m4['ewma4_x_due'] = m4['ewma_4']*m4['due_ratio']
r4 = ev(m4); print('v4 +interactions: %.2f / %.2f  d %+.2f'%(r4[0],r4[1],r4[1]-b[1]))
p = agent_api.save_table(m4, 'e017_hazard_ext.parquet'); print('saved', p)

# ---- cell ----
import agent_api, pandas as pd, numpy as np
base = agent_api.load_saved('e009_ewma_longlags.parquet')
m2 = agent_api.load_saved('e017_season_hazard.parquet')
tt = agent_api.train_targets()
def ev(df, alpha=100.0):
    df = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
    cols=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()
b = ev(base)
haz = m2.drop(columns=['hh_season_next4','hh_season_next4_x','glob_season_next4'])
rh = ev(haz); print('hazard-only (4 feats): %.2f / %.2f  d %+.2f'%(rh[0],rh[1],rh[1]-b[1]))
r2 = ev(m2); print('full v2 (7 feats):     %.2f / %.2f  d %+.2f'%(r2[0],r2[1],r2[1]-b[1]))
p = agent_api.save_table(haz, 'e017_hazard_only.parquet'); print('saved', p)