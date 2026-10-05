import pandas as pd, numpy as np
import agent_api

e = agent_api.load_saved('e011_table.parquet')
print('e011_table shape', e.shape)
cols = list(e.columns)
print('COLS:', ', '.join(cols))
print()

for name in ['selfcal_v1','churn_vol_v1','season_demo_v1','rfm_cadence_v1','rfm_traj_v1','ewma_block_v1','mkt_v1','comp_v1','demo_v1','rfm_v1']:
    try:
        d = agent_api.load_saved(name+'.parquet')
        cc = [c for c in d.columns if c not in ('household_key','snapshot_day')]
        print(name, d.shape, '|', ', '.join(cc[:80]))
    except Exception as ex:
        print(name, 'ERR', ex)
print()

tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share overall', float((tt.future_spend_4w==0).mean()))
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median'])
g['zero'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda s: float((s==0).mean()))
print(g)

def ridge_eval(df, cols, holdouts=(403,431), lam=30.0):
    cols = list(cols)
    d = tt.merge(df[['household_key','snapshot_day']+cols], on=['household_key','snapshot_day'], how='inner')
    X = d[cols].apply(pd.to_numeric, errors='coerce').values.astype(float)
    y = d['future_spend_4w'].values.astype(float)
    days = d['snapshot_day'].values
    out=[]
    for h in holdouts:
        tr = days!=h; te = days==h
        mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0)
        sd[~np.isfinite(sd)|(sd==0)]=1.0
        A = np.where(np.isnan(X[tr]),0.0,(X[tr]-mu)/sd)
        B = np.where(np.isnan(X[te]),0.0,(X[te]-mu)/sd)
        w = np.linalg.solve(A.T@A + lam*np.eye(len(cols)), A.T@y[tr])
        p = B@w
        out.append(float(np.mean(np.abs(p-y[te]))))
    return float(np.mean(out)), out

num_cols = [c for c in cols if c not in ('household_key','snapshot_day')]
print()
print('ridge all e011:', ridge_eval(e, num_cols))


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api

e = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()

for name in ['mkt_v1','comp_v1']:
    d = agent_api.load_saved(name+'.parquet')
    cc = [c for c in d.columns if c not in ('household_key','snapshot_day')]
    print(name, len(cc)); print(' ', ', '.join(cc)); print()

NB = 34
def make_edges(Xtr):
    edges=[]
    for f in range(Xtr.shape[1]):
        v = Xtr[:,f]; v = v[np.isfinite(v)]
        edges.append(np.unique(np.quantile(v, np.linspace(0,1,33)[1:-1])) if len(v) else None)
    return edges

def apply_bins(X, edges):
    B = np.zeros(X.shape, dtype=np.uint8)
    for f in range(X.shape[1]):
        qs = edges[f]
        if qs is None: continue
        col = X[:,f]
        b = np.searchsorted(qs, col, side='right')
        b[~np.isfinite(col)] = len(qs)+1
        B[:,f] = b.astype(np.uint8)
    return B

def gbm(X, y, days, val_day=431, trees=250, lr=0.08, depth=3, lam=1.0, mc=20, tag=''):
    tr = days < val_day; te = days == val_day
    edges = make_edges(X[tr]); Btr = apply_bins(X[tr], edges); Bte = apply_bins(X[te], edges)
    base = y[tr].mean(); p = np.full(tr.sum(), base); pt = np.full(te.sum(), base)
    gains = np.zeros(X.shape[1]); t0=time.time()
    for t in range(trees):
        g = p - y[tr]
        nodes = {0:{}}; queue=[(np.arange(tr.sum()),0,0)]; i=0
        while i < len(queue):
            idx, dep, nid = queue[i]; i+=1; m=len(idx)
            if dep>=depth or m<2*mc:
                nodes[nid]['leaf'] = -g[idx].sum()/(m+lam); continue
            sub = Btr[idx]; G = g[idx]; best=(0.0,-1,-1)
            for f in range(X.shape[1]):
                bh = np.bincount(sub[:,f], minlength=NB).astype(np.float64)
                bg = np.bincount(sub[:,f], weights=G, minlength=NB)
                ch = np.cumsum(bh); cg = np.cumsum(bg)
                GL,HL = cg[:-1],ch[:-1]; GR,HR = cg[-1]-GL, ch[-1]-HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - (GL+GR)**2/(HL+HR+lam)
                k = int(np.argmax(gain))
                if gain[k]>best[0] and ch[k]>=mc and (ch[-1]-ch[k])>=mc: best=(float(gain[k]),f,k)
            if best[1]<0:
                nodes[nid]['leaf'] = -g[idx].sum()/(m+lam); continue
            gv,f,thr = best; gains[f]+=gv
            mask = sub[:,f]<=thr; L,R = idx[mask], idx[~mask]
            nodes[nid]['split']=(f,thr); lid=len(nodes); nodes[lid]={}; rid=len(nodes); nodes[rid]={}
            nodes[nid]['left'],nodes[nid]['right']=lid,rid
            queue.append((L,dep+1,lid)); queue.append((R,dep+1,rid))
        leafmap = {n_:nd['leaf'] for n_,nd in nodes.items() if 'leaf' in nd}
        cur = np.zeros(tr.sum(),dtype=np.int32); curt = np.zeros(te.sum(),dtype=np.int32)
        for nid in sorted(nodes):
            nd = nodes[nid]
            if 'split' in nd:
                f,thr = nd['split']
                s = cur==nid;  cur[s & (Btr[:,f]<=thr)]  = nd['left']; cur[s & (Btr[:,f]>thr)]  = nd['right']
                s = curt==nid; curt[s & (Bte[:,f]<=thr)] = nd['left']; curt[s & (Bte[:,f]>thr)] = nd['right']
        p  += lr*np.array([leafmap[c] for c in cur])
        pt += lr*np.array([leafmap.get(c,0.0) for c in curt])
        if (t+1)%100==0: print(f'  {tag} tree {t+1} val431 MAE {np.mean(np.abs(pt-y[te])):.3f} ({time.time()-t0:.0f}s)')
    print(f'GBM[{tag}] final val431 MAE {np.mean(np.abs(pt-y[te])):.3f}  ({time.time()-t0:.0f}s)')
    return np.mean(np.abs(pt-y[te])), gains, pt, y[te]

def ridge(X, y, days, val_day=431, lam=30.0):
    tr = days<val_day; te = days==val_day
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[~np.isfinite(sd)|(sd==0)]=1
    A = np.where(np.isnan(X[tr]),0,(X[tr]-mu)/sd); B = np.where(np.isnan(X[te]),0,(X[te]-mu)/sd)
    w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@y[tr])
    return float(np.mean(np.abs(B@w - y[te])))

d = tt.merge(e, on=['household_key','snapshot_day'], how='inner')
y = d['future_spend_4w'].values.astype(float); days = d['snapshot_day'].values
allc = [c for c in e.columns if c not in ('household_key','snapshot_day')]

e001 = ['spend_28d','trips_28d','spend_56d','trips_56d','spend_84d','trips_84d','spend_112d','trips_112d',
        'spend_168d','trips_168d','spend_364d','trips_364d','days_since_last','avg_basket_112','trend_28',
        'spend_per_week_84','active_weeks_112']

X1 = d[e001].values.astype(float)
print('E001 feats: ridge', round(ridge(X1,y,days),3), '(harness 61.532)')
gbm(X1, y, days, trees=250, tag='E001')
Xs = d[['spend_28d']].values.astype(float)
print('spend_28d only: ridge', round(ridge(Xs,y,days),3))
gbm(Xs, y, days, trees=250, tag='s28')


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api

e = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
NB = 34
def make_edges(Xtr):
    edges=[]
    for f in range(Xtr.shape[1]):
        v = Xtr[:,f]; v = v[np.isfinite(v)]
        edges.append(np.unique(np.quantile(v, np.linspace(0,1,33)[1:-1])) if len(v) else None)
    return edges
def apply_bins(X, edges):
    B = np.zeros(X.shape, dtype=np.uint8)
    for f in range(X.shape[1]):
        qs = edges[f]
        if qs is None: continue
        col = X[:,f]
        b = np.searchsorted(qs, col, side='right')
        b[~np.isfinite(col)] = len(qs)+1
        B[:,f] = b.astype(np.uint8)
    return B
def gbm(X, y, days, val_day=431, trees=220, lr=0.08, depth=3, lam=1.0, mc=20, tag='', verbose=False):
    tr = days < val_day; te = days == val_day
    edges = make_edges(X[tr]); Btr = apply_bins(X[tr], edges); Bte = apply_bins(X[te], edges)
    base = y[tr].mean(); p = np.full(tr.sum(), base); pt = np.full(te.sum(), base)
    gains = np.zeros(X.shape[1]); t0=time.time()
    for t in range(trees):
        g = p - y[tr]
        nodes = {0:{}}; queue=[(np.arange(tr.sum()),0,0)]; i=0
        while i < len(queue):
            idx, dep, nid = queue[i]; i+=1; m=len(idx)
            if dep>=depth or m<2*mc:
                nodes[nid]['leaf'] = -g[idx].sum()/(m+lam); continue
            sub = Btr[idx]; G = g[idx]; best=(0.0,-1,-1)
            for f in range(X.shape[1]):
                bh = np.bincount(sub[:,f], minlength=NB).astype(np.float64)
                bg = np.bincount(sub[:,f], weights=G, minlength=NB)
                ch = np.cumsum(bh); cg = np.cumsum(bg)
                GL,HL = cg[:-1],ch[:-1]; GR,HR = cg[-1]-GL, ch[-1]-HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - (GL+GR)**2/(HL+HR+lam)
                k = int(np.argmax(gain))
                if gain[k]>best[0] and ch[k]>=mc and (ch[-1]-ch[k])>=mc: best=(float(gain[k]),f,k)
            if best[1]<0:
                nodes[nid]['leaf'] = -g[idx].sum()/(m+lam); continue
            gv,f,thr = best; gains[f]+=gv
            mask = sub[:,f]<=thr; L,R = idx[mask], idx[~mask]
            nodes[nid]['split']=(f,thr); lid=len(nodes); nodes[lid]={}; rid=len(nodes); nodes[rid]={}
            nodes[nid]['left'],nodes[nid]['right']=lid,rid
            queue.append((L,dep+1,lid)); queue.append((R,dep+1,rid))
        leafmap = {n_:nd['leaf'] for n_,nd in nodes.items() if 'leaf' in nd}
        cur = np.zeros(tr.sum(),dtype=np.int32); curt = np.zeros(te.sum(),dtype=np.int32)
        for nid in sorted(nodes):
            nd = nodes[nid]
            if 'split' in nd:
                f,thr = nd['split']
                s = cur==nid;  cur[s & (Btr[:,f]<=thr)]  = nd['left']; cur[s & (Btr[:,f]>thr)]  = nd['right']
                s = curt==nid; curt[s & (Bte[:,f]<=thr)] = nd['left']; curt[s & (Bte[:,f]>thr)] = nd['right']
        p  += lr*np.array([leafmap[c] for c in cur])
        pt += lr*np.array([leafmap.get(c,0.0) for c in curt])
    mae = float(np.mean(np.abs(pt-y[te])))
    if verbose: print(f'GBM[{tag}] val431 MAE {mae:.3f} ({time.time()-t0:.0f}s)')
    return mae, gains

def ridge(X, y, days, val_day=431, lam=30.0):
    tr = days<val_day; te = days==val_day
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[~np.isfinite(sd)|(sd==0)]=1
    A = np.where(np.isnan(X[tr]),0,(X[tr]-mu)/sd); B = np.where(np.isnan(X[te]),0,(X[te]-mu)/sd)
    w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@y[tr])
    return float(np.mean(np.abs(B@w - y[te])))

d = tt.merge(e, on=['household_key','snapshot_day'], how='inner')
y = d['future_spend_4w'].values.astype(float); days = d['snapshot_day'].values
allc = [c for c in e.columns if c not in ('household_key','snapshot_day')]
X = d[allc].values.astype(float)
print('E011 all feats: ridge', round(ridge(X,y,days),3))
mae0, gains = gbm(X, y, days, tag='E011', verbose=True)
top = np.argsort(-gains)[:40]
print('top gain feats:', [allc[i] for i in top])

Xc = np.column_stack([X, days.astype(float), ((days-95)/28).astype(float)])
print('E011 + snapday: ridge', round(ridge(Xc,y,days),3), '| gbm', round(gbm(Xc,y,days,tag='drift')[0],3))

df = d[['household_key','snapshot_day','future_spend_4w']+allc].copy()
for c in ['spend_28d','spend_84d','spend_364d','ew_spend_hl28','ew_spend_hl56','usual_4w','sc_f_mean','ya_spend','b_life_spend_pw']:
    if c in df.columns:
        g = df.groupby('snapshot_day')[c]
        df[c+'_rank'] = g.rank(pct=True)
        df[c+'_z'] = (df[c]-g.transform('mean'))/g.transform('std').replace(0,np.nan)
rn = [c for c in df.columns if c.endswith('_rank') or c.endswith('_z')]
Xc2 = df[rn].values.astype(float)
print('ranks only: ridge', round(ridge(Xc2,y,df.snapshot_day.values),3), '| gbm', round(gbm(Xc2,y,df.snapshot_day.values,tag='ranks')[0],3))
Xc3 = np.column_stack([X, df[rn].values.astype(float)])
print('E011 + ranks/z: ridge', round(ridge(Xc3,y,days),3), '| gbm', round(gbm(Xc3,y,days,tag='E011+ranks')[0],3))


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api

e = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
d = tt.merge(e, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
y = d['future_spend_4w'].values.astype(float); days = d['snapshot_day'].values
hh = d['household_key'].values
allc = [c for c in e.columns if c not in ('household_key','snapshot_day')]

snap = agent_api.snapshot(459)
tx = snap.transactions
g = tx.groupby('household_key')
hh_days = {}; hh_sales = {}; hh_first = {}
for k, idx in g.indices.items():
    dd = tx['day'].values[idx]; ss = tx['sales_value'].values[idx]
    o = np.argsort(dd, kind='stable'); dd = dd[o]; ss = ss[o]
    hh_days[k] = dd; hh_sales[k] = ss; hh_first[k] = dd[0]
first_day = pd.Series(hh_first)

def wsum(k, lo, hi):
    dd = hh_days.get(k)
    if dd is None: return 0.0
    a = np.searchsorted(dd, lo, 'left'); b = np.searchsorted(dd, hi, 'right')
    return float(hh_sales[k][a:b].sum())
def wcnt(k, lo, hi):
    dd = hh_days.get(k)
    if dd is None: return 0
    a = np.searchsorted(dd, lo, 'left'); b = np.searchsorted(dd, hi, 'right')
    return float(b-a)

S = d['snapshot_day'].values
A = pd.DataFrame(index=d.index)
A['spend_7d']  = [wsum(k, s-6, s)  for k,s in zip(hh,S)]
A['spend_14d'] = [wsum(k, s-13, s) for k,s in zip(hh,S)]
A['spend_21d'] = [wsum(k, s-20, s) for k,s in zip(hh,S)]
A['trips_7d']  = [wcnt(k, s-6, s)  for k,s in zip(hh,S)]
A['trips_14d'] = [wcnt(k, s-13, s) for k,s in zip(hh,S)]
A['last_basket'] = [ (hh_sales[k][-1] if len(hh_days.get(k,[])) else 0.0) for k in hh]
A['phase'] = d['days_since_last'].values/28.0
A['r7_28'] = A['spend_7d']/(d['spend_28d'].values+1.0)
A['r14_28'] = A['spend_14d']/(d['spend_28d'].values+1.0)
A['ew_trips_hl28'] = [ (lambda dd: float(np.sum(np.exp(-np.log(2)*(s-dd)/28))) if len(dd) else 0.0)(hh_days.get(k, np.array([]))) for k,s in zip(hh,S)]
acols = list(A.columns); A = A.astype(float)

elig = (first_day.reindex(hh).values <= (S-84))
B = pd.DataFrame(index=d.index)
for k in [1,2,4]:
    lo, hi = S-28*k+1, S-28*k+28
    vals = np.array([wsum(kk, a, b) for kk,a,b in zip(hh,lo,hi)])
    v = np.where(elig, vals, np.nan)
    B[f'mkt_f_mean_{k}'] = pd.Series(v).groupby(S).transform('mean').values
    B[f'mkt_f_med_{k}'] = pd.Series(v).groupby(S).transform('median').values
B['mkt_t_mean'] = d.groupby('snapshot_day')['spend_28d'].transform('mean').values
B['rel_28'] = d['spend_28d'].values/(B['mkt_t_mean'].values+1e-9)
B['rel_28_f1'] = d['spend_28d'].values/(B['mkt_f_mean_1'].values+1e-9)
bcols = list(B.columns); B = B.astype(float)

C = pd.DataFrame(index=d.index)
C['x_28_ratio'] = d['spend_28d'].values * d['sc_ratio_mean'].fillna(1.0).values
C['x_ew28_ratio'] = d['ew_spend_hl28'].values * d['sc_ratio_mean'].fillna(1.0).values
C['x_28_zerof'] = d['spend_28d'].values * (1-d['sc_f_zero_frac'].fillna(0.2).values)
C['x_usual_ratio'] = d['usual_4w'].values * d['sc_ratio_mean'].fillna(1.0).values
ccols = list(C.columns); C = C.astype(float)

NB = 34
def make_edges(Xtr):
    edges=[]
    for f in range(Xtr.shape[1]):
        v = Xtr[:,f]; v = v[np.isfinite(v)]
        edges.append(np.unique(np.quantile(v, np.linspace(0,1,33)[1:-1])) if len(v) else None)
    return edges
def apply_bins(X, edges):
    Bx = np.zeros(X.shape, dtype=np.uint8)
    for f in range(X.shape[1]):
        qs = edges[f]
        if qs is None: continue
        col = X[:,f]
        b = np.searchsorted(qs, col, side='right')
        b[~np.isfinite(col)] = len(qs)+1
        Bx[:,f] = b.astype(np.uint8)
    return Bx
def gbm(X, y, days, val_day=431, trees=180, lr=0.08, depth=3, lam=1.0, mc=20, tag=''):
    tr = days < val_day; te = days == val_day
    edges = make_edges(X[tr]); Btr = apply_bins(X[tr], edges); Bte = apply_bins(X[te], edges)
    base = y[tr].mean(); p = np.full(tr.sum(), base); pt = np.full(te.sum(), base)
    gains = np.zeros(X.shape[1])
    for t in range(trees):
        gr = p - y[tr]
        nodes = {0:{}}; queue=[(np.arange(tr.sum()),0,0)]; i=0
        while i < len(queue):
            idx, dep, nid = queue[i]; i+=1; m=len(idx)
            if dep>=depth or m<2*mc:
                nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            sub = Btr[idx]; G = gr[idx]; best=(0.0,-1,-1)
            for f in range(X.shape[1]):
                bh = np.bincount(sub[:,f], minlength=NB).astype(np.float64)
                bg = np.bincount(sub[:,f], weights=G, minlength=NB)
                ch = np.cumsum(bh); cg = np.cumsum(bg)
                GL,HL = cg[:-1],ch[:-1]; GR,HR = cg[-1]-GL, ch[-1]-HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - (GL+GR)**2/(HL+HR+lam)
                kk = int(np.argmax(gain))
                if gain[kk]>best[0] and ch[kk]>=mc and (ch[-1]-ch[kk])>=mc: best=(float(gain[kk]),f,kk)
            if best[1]<0:
                nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            gv,f,thr = best; gains[f]+=gv
            mask = sub[:,f]<=thr; L,R = idx[mask], idx[~mask]
            nodes[nid]['split']=(f,thr); lid=len(nodes); nodes[lid]={}; rid=len(nodes); nodes[rid]={}
            nodes[nid]['left'],nodes[nid]['right']=lid,rid
            queue.append((L,dep+1,lid)); queue.append((R,dep+1,rid))
        leafmap = {n_:nd['leaf'] for n_,nd in nodes.items() if 'leaf' in nd}
        cur = np.zeros(tr.sum(),dtype=np.int32); curt = np.zeros(te.sum(),dtype=np.int32)
        for nid in sorted(nodes):
            nd = nodes[nid]
            if 'split' in nd:
                f,thr = nd['split']
                s_ = cur==nid;  cur[s_ & (Btr[:,f]<=thr)]  = nd['left']; cur[s_ & (Btr[:,f]>thr)]  = nd['right']
                s_ = curt==nid; curt[s_ & (Bte[:,f]<=thr)] = nd['left']; curt[s_ & (Bte[:,f]>thr)] = nd['right']
        p  += lr*np.array([leafmap[c] for c in cur])
        pt += lr*np.array([leafmap.get(c,0.0) for c in curt])
    mae = float(np.mean(np.abs(pt-y[te])))
    print(f'GBM[{tag}] val431 MAE {mae:.3f}')
    return mae, gains, pt, y[te]

X0 = d[allc].values.astype(float)
mae0, gains0, p0, y0 = gbm(X0, y, days, tag='E011 base')
gbm(np.column_stack([X0, A.values]), y, days, tag='E011+fine')
gbm(np.column_stack([X0, B.values]), y, days, tag='E011+mkt')
gbm(np.column_stack([X0, C.values]), y, days, tag='E011+inter')
gbm(np.column_stack([X0, A.values, B.values, C.values]), y, days, tag='E011+ABC')
top40 = [allc[i] for i in np.argsort(-gains0)[:40]]
Xl = d[top40].values.astype(float)
gbm(Xl, y, days, tag='lean40')
gbm(np.column_stack([Xl, A.values, B.values, C.values]), y, days, tag='lean40+ABC')

res = p0 - y0
dec = pd.qcut(pd.Series(d['spend_28d'].values)[days==431], 10, duplicates='drop')
diag = pd.DataFrame({'y':y0,'p':p0,'res':res,'dec':dec.values,'zero':(y0==0),
                     'dsl':d['days_since_last'].values[days==431],'scz':d['sc_f_zero_frac'].values[days==431]})
print('\nby spend_28d decile:')
print(diag.groupby('dec', observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g.y.mean(),'p':g.p.mean(),'bias':g.res.mean(),'mae':np.abs(g.res).mean()})))
print('\nzero-actual rows: n, mean p, MAE:', len(diag[diag.zero]), round(diag[diag.zero].p.mean(),2), round(np.abs(diag[diag.zero].res).mean(),2))
print('active rows bias/MAE:', round(diag[~diag.zero].res.mean(),2), round(np.abs(diag[~diag.zero].res).mean(),2))
print('overall bias (p-y):', round(res.mean(),2), ' MAE:', round(np.abs(res).mean(),2))


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
e006 = agent_api.load_saved('demo_v1.parquet')
tt = agent_api.train_targets()

snap = agent_api.snapshot(459)
tx = snap.transactions
g = tx.groupby('household_key')
hh_days = {}; hh_sales = {}
for k, idx in g.indices.items():
    dd = tx['day'].values[idx]; ss = tx['sales_value'].values[idx]
    o = np.argsort(dd, kind='stable'); dd = dd[o]; ss = ss[o]
    hh_days[k] = dd; hh_sales[k] = ss

def wsum(k, lo, hi):
    dd = hh_days.get(k)
    if dd is None: return 0.0
    a = np.searchsorted(dd, lo, 'left'); b = np.searchsorted(dd, hi, 'right')
    return float(hh_sales[k][a:b].sum())
def wcnt(k, lo, hi):
    dd = hh_days.get(k)
    if dd is None: return 0
    a = np.searchsorted(dd, lo, 'left'); b = np.searchsorted(dd, hi, 'right')
    return float(b-a)

def build_fine(df):
    hh = df['household_key'].values; S = df['snapshot_day'].values
    F = pd.DataFrame(index=df.index)
    F['spend_7d']  = [wsum(k, s-6, s)  for k,s in zip(hh,S)]
    F['spend_14d'] = [wsum(k, s-13, s) for k,s in zip(hh,S)]
    F['spend_21d'] = [wsum(k, s-20, s) for k,s in zip(hh,S)]
    F['trips_7d']  = [wcnt(k, s-6, s)  for k,s in zip(hh,S)]
    F['trips_14d'] = [wcnt(k, s-13, s) for k,s in zip(hh,S)]
    F['last_basket'] = [(hh_sales[k][-1] if len(hh_days.get(k,[])) else 0.0) for k in hh]
    F['phase'] = df['days_since_last'].values/28.0
    F['r7_28'] = F['spend_7d']/(df['spend_28d'].values+1.0)
    F['r14_28'] = F['spend_14d']/(df['spend_28d'].values+1.0)
    F['ew_trips_hl28'] = [(float(np.sum(np.exp(-np.log(2)*(s-dd)/28))) if len(dd) else 0.0) for dd,s in zip([hh_days.get(k,np.array([])) for k in hh],S)]
    return F.astype(float)

NB = 34
def make_edges(Xtr):
    edges=[]
    for f in range(Xtr.shape[1]):
        v = Xtr[:,f]; v = v[np.isfinite(v)]
        edges.append(np.unique(np.quantile(v, np.linspace(0,1,33)[1:-1])) if len(v) else None)
    return edges
def apply_bins(X, edges):
    Bx = np.zeros(X.shape, dtype=np.uint8)
    for f in range(X.shape[1]):
        qs = edges[f]
        if qs is None: continue
        col = X[:,f]
        b = np.searchsorted(qs, col, side='right')
        b[~np.isfinite(col)] = len(qs)+1
        Bx[:,f] = b.astype(np.uint8)
    return Bx
def gbm(df, extra, trees=180, lr=0.08, depth=3, lam=1.0, mc=20, tag=''):
    dd = tt.merge(df, on=['household_key','snapshot_day'], how='inner')
    y = dd['future_spend_4w'].values.astype(float); days = dd['snapshot_day'].values
    base = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    X = np.column_stack([dd[base].values.astype(float)] + ([extra.values.astype(float)] if extra is not None and len(extra) else []))
    tr = days < 431; te = days == 431
    edges = make_edges(X[tr]); Btr = apply_bins(X[tr], edges); Bte = apply_bins(X[te], edges)
    b0 = y[tr].mean(); p = np.full(tr.sum(), b0); pt = np.full(te.sum(), b0)
    for t in range(trees):
        gr = p - y[tr]
        nodes = {0:{}}; queue=[(np.arange(tr.sum()),0,0)]; i=0
        while i < len(queue):
            idx, dep, nid = queue[i]; i+=1; m=len(idx)
            if dep>=depth or m<2*mc:
                nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            sub = Btr[idx]; G = gr[idx]; best=(0.0,-1,-1)
            for f in range(X.shape[1]):
                bh = np.bincount(sub[:,f], minlength=NB).astype(np.float64)
                bg = np.bincount(sub[:,f], weights=G, minlength=NB)
                ch = np.cumsum(bh); cg = np.cumsum(bg)
                GL,HL = cg[:-1],ch[:-1]; GR,HR = cg[-1]-GL, ch[-1]-HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - (GL+GR)**2/(HL+HR+lam)
                kk = int(np.argmax(gain))
                if gain[kk]>best[0] and ch[kk]>=mc and (ch[-1]-ch[kk])>=mc: best=(float(gain[kk]),f,kk)
            if best[1]<0:
                nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            gv,f,thr = best
            mask = sub[:,f]<=thr; L,R = idx[mask], idx[~mask]
            nodes[nid]['split']=(f,thr); lid=len(nodes); nodes[lid]={}; rid=len(nodes); nodes[rid]={}
            nodes[nid]['left'],nodes[nid]['right']=lid,rid
            queue.append((L,dep+1,lid)); queue.append((R,dep+1,rid))
        leafmap = {n_:nd['leaf'] for n_,nd in nodes.items() if 'leaf' in nd}
        cur = np.zeros(tr.sum(),dtype=np.int32); curt = np.zeros(te.sum(),dtype=np.int32)
        for nid in sorted(nodes):
            nd = nodes[nid]
            if 'split' in nd:
                f,thr = nd['split']
                s_ = cur==nid;  cur[s_ & (Btr[:,f]<=thr)]  = nd['left']; cur[s_ & (Btr[:,f]>thr)]  = nd['right']
                s_ = curt==nid; curt[s_ & (Bte[:,f]<=thr)] = nd['left']; curt[s_ & (Bte[:,f]>thr)] = nd['right']
        p  += lr*np.array([leafmap[c] for c in cur])
        pt += lr*np.array([leafmap.get(c,0.0) for c in curt])
    mae = float(np.mean(np.abs(pt-y[te])))
    print(f'GBM[{tag}] val431 MAE {mae:.3f}')
    return mae

# fidelity check: fine block on E006 table (harness E007 said ~neutral)
f006 = build_fine(e006)
gbm(e006, None, tag='E006 base')
gbm(e006, f006, tag='E006+fine')

# ablation on E011
f011 = build_fine(e011)
gbm(e011, None, tag='E011 base')
for subset, name in [
    (['spend_7d','spend_14d','spend_21d'],'windows only'),
    (['trips_7d','trips_14d','ew_trips_hl28'],'trips recency only'),
    (['last_basket','phase'],'lastbasket+phase only'),
    (['r7_28','r14_28'],'ratios only'),
    (['spend_7d','spend_14d','spend_21d','trips_7d','trips_14d','ew_trips_hl28'],'windows+trips'),
]:
    sub = f011[subset]
    gbm(e011, sub, tag=f'E011+{name}')
gbm(e011, f011, tag='E011+fine (all 10)')


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
e006 = agent_api.load_saved('demo_v1.parquet')
tt = agent_api.train_targets()

snap = agent_api.snapshot(459)
tx = snap.transactions
g = tx.groupby('household_key')
hh_days = {}; hh_sales = {}
for k, idx in g.indices.items():
    dd = tx['day'].values[idx]; ss = tx['sales_value'].values[idx]
    o = np.argsort(dd, kind='stable'); dd = dd[o]; ss = ss[o]
    hh_days[k] = dd; hh_sales[k] = ss

def wsum(k, lo, hi):
    dd = hh_days.get(k)
    if dd is None: return 0.0
    a = np.searchsorted(dd, lo, 'left'); b = np.searchsorted(dd, hi, 'right')
    return float(hh_sales[k][a:b].sum())
def wcnt(k, lo, hi):
    dd = hh_days.get(k)
    if dd is None: return 0
    a = np.searchsorted(dd, lo, 'left'); b = np.searchsorted(dd, hi, 'right')
    return float(b-a)

def build_fine(df):
    hh = df['household_key'].values; S = df['snapshot_day'].values
    F = pd.DataFrame(index=df.index)
    F['spend_7d']  = [wsum(k, s-6, s)  for k,s in zip(hh,S)]
    F['spend_14d'] = [wsum(k, s-13, s) for k,s in zip(hh,S)]
    F['spend_21d'] = [wsum(k, s-20, s) for k,s in zip(hh,S)]
    F['trips_7d']  = [wcnt(k, s-6, s)  for k,s in zip(hh,S)]
    F['trips_14d'] = [wcnt(k, s-13, s) for k,s in zip(hh,S)]
    F['last_basket'] = [(hh_sales[k][-1] if len(hh_days.get(k,[])) else 0.0) for k in hh]
    F['phase'] = df['days_since_last'].values/28.0
    F['r7_28'] = F['spend_7d']/(df['spend_28d'].values+1.0)
    F['r14_28'] = F['spend_14d']/(df['spend_28d'].values+1.0)
    F['ew_trips_hl28'] = [(float(np.sum(np.exp(-np.log(2)*(s-dd)/28))) if len(dd) else 0.0) for dd,s in zip([hh_days.get(k,np.array([])) for k in hh],S)]
    return F.astype(float)

NB = 34
def make_edges(Xtr):
    edges=[]
    for f in range(Xtr.shape[1]):
        v = Xtr[:,f]; v = v[np.isfinite(v)]
        edges.append(np.unique(np.quantile(v, np.linspace(0,1,33)[1:-1])) if len(v) else None)
    return edges
def apply_bins(X, edges):
    Bx = np.zeros(X.shape, dtype=np.uint8)
    for f in range(X.shape[1]):
        qs = edges[f]
        if qs is None: continue
        col = X[:,f]
        b = np.searchsorted(qs, col, side='right')
        b[~np.isfinite(col)] = len(qs)+1
        Bx[:,f] = b.astype(np.uint8)
    return Bx
def gbm(df, trees=180, lr=0.08, depth=3, lam=1.0, mc=20, tag=''):
    dd = tt.merge(df, on=['household_key','snapshot_day'], how='inner')
    y = dd['future_spend_4w'].values.astype(float); days = dd['snapshot_day'].values
    base = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    X = dd[base].values.astype(float)
    tr = days < 431; te = days == 431
    edges = make_edges(X[tr]); Btr = apply_bins(X[tr], edges); Bte = apply_bins(X[te], edges)
    b0 = y[tr].mean(); p = np.full(tr.sum(), b0); pt = np.full(te.sum(), b0)
    for t in range(trees):
        gr = p - y[tr]
        nodes = {0:{}}; queue=[(np.arange(tr.sum()),0,0)]; i=0
        while i < len(queue):
            idx, dep, nid = queue[i]; i+=1; m=len(idx)
            if dep>=depth or m<2*mc:
                nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            sub = Btr[idx]; G = gr[idx]; best=(0.0,-1,-1)
            for f in range(X.shape[1]):
                bh = np.bincount(sub[:,f], minlength=NB).astype(np.float64)
                bg = np.bincount(sub[:,f], weights=G, minlength=NB)
                ch = np.cumsum(bh); cg = np.cumsum(bg)
                GL,HL = cg[:-1],ch[:-1]; GR,HR = cg[-1]-GL, ch[-1]-HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - (GL+GR)**2/(HL+HR+lam)
                kk = int(np.argmax(gain))
                if gain[kk]>best[0] and ch[kk]>=mc and (ch[-1]-ch[kk])>=mc: best=(float(gain[kk]),f,kk)
            if best[1]<0:
                nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            gv,f,thr = best
            mask = sub[:,f]<=thr; L,R = idx[mask], idx[~mask]
            nodes[nid]['split']=(f,thr); lid=len(nodes); nodes[lid]={}; rid=len(nodes); nodes[rid]={}
            nodes[nid]['left'],nodes[nid]['right']=lid,rid
            queue.append((L,dep+1,lid)); queue.append((R,dep+1,rid))
        leafmap = {n_:nd['leaf'] for n_,nd in nodes.items() if 'leaf' in nd}
        cur = np.zeros(tr.sum(),dtype=np.int32); curt = np.zeros(te.sum(),dtype=np.int32)
        for nid in sorted(nodes):
            nd = nodes[nid]
            if 'split' in nd:
                f,thr = nd['split']
                s_ = cur==nid;  cur[s_ & (Btr[:,f]<=thr)]  = nd['left']; cur[s_ & (Btr[:,f]>thr)]  = nd['right']
                s_ = curt==nid; curt[s_ & (Bte[:,f]<=thr)] = nd['left']; curt[s_ & (Bte[:,f]>thr)] = nd['right']
        p  += lr*np.array([leafmap[c] for c in cur])
        pt += lr*np.array([leafmap.get(c,0.0) for c in curt])
    mae = float(np.mean(np.abs(pt-y[te])))
    print(f'GBM[{tag}] val431 MAE {mae:.3f}')
    return mae

def with_extra(df, extra):
    return df.merge(extra, left_index=True, right_index=True, how='left')

# fidelity check: fine block on E006 table (harness E007 said ~neutral)
f006 = build_fine(e006)
gbm(e006, tag='E006 base')
gbm(with_extra(e006, f006), tag='E006+fine')

# ablation on E011
f011 = build_fine(e011)
gbm(e011, tag='E011 base')
for subset, name in [
    (['spend_7d','spend_14d','spend_21d'],'windows only'),
    (['trips_7d','trips_14d','ew_trips_hl28'],'trips recency only'),
    (['last_basket','phase'],'lastbasket+phase only'),
    (['r7_28','r14_28'],'ratios only'),
    (['spend_7d','spend_14d','spend_21d','trips_7d','trips_14d','ew_trips_hl28'],'windows+trips'),
]:
    gbm(with_extra(e011, f011[subset]), tag=f'E011+{name}')
gbm(with_extra(e011, f011), tag='E011+fine (all 10)')


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
e006 = agent_api.load_saved('demo_v1.parquet')
tt = agent_api.train_targets()

snap = agent_api.snapshot(459)
tx = snap.transactions
g = tx.groupby('household_key')
hh_days = {}; hh_sales = {}
for k, idx in g.indices.items():
    dd = tx['day'].values[idx]; ss = tx['sales_value'].values[idx]
    o = np.argsort(dd, kind='stable'); dd = dd[o]; ss = ss[o]
    hh_days[k] = dd; hh_sales[k] = ss

def wsum(k, lo, hi):
    dd = hh_days.get(k)
    if dd is None: return 0.0
    a = np.searchsorted(dd, lo, 'left'); b = np.searchsorted(dd, hi, 'right')
    return float(hh_sales[k][a:b].sum())
def wcnt(k, lo, hi):
    dd = hh_days.get(k)
    if dd is None: return 0
    a = np.searchsorted(dd, lo, 'left'); b = np.searchsorted(dd, hi, 'right')
    return float(b-a)

def build_fine(df):  # STRICTLY causal: only days <= s
    hh = df['household_key'].values; S = df['snapshot_day'].values
    F = pd.DataFrame(index=df.index)
    F['spend_7d']  = [wsum(k, s-6, s)  for k,s in zip(hh,S)]
    F['spend_14d'] = [wsum(k, s-13, s) for k,s in zip(hh,S)]
    F['spend_21d'] = [wsum(k, s-20, s) for k,s in zip(hh,S)]
    F['trips_7d']  = [wcnt(k, s-6, s)  for k,s in zip(hh,S)]
    F['trips_14d'] = [wcnt(k, s-13, s) for k,s in zip(hh,S)]
    F['last_basket'] = [(hh_sales[k][np.searchsorted(hh_days[k], s, 'right')-1] if len(hh_days.get(k,[])) and hh_days[k][0]<=s else 0.0) for k,s in zip(hh,S)]
    F['phase'] = df['days_since_last'].values/28.0
    F['r7_28'] = F['spend_7d']/(df['spend_28d'].values+1.0)
    F['r14_28'] = F['spend_14d']/(df['spend_28d'].values+1.0)
    F['ew_trips_hl28'] = [(float(np.sum(np.exp(-np.log(2)*(s-dd[dd<=s])/28))) if len(dd) else 0.0) for dd,s in zip([hh_days.get(k,np.array([])) for k in hh],S)]
    return F.astype(float)

NB = 34
def make_edges(Xtr):
    edges=[]
    for f in range(Xtr.shape[1]):
        v = Xtr[:,f]; v = v[np.isfinite(v)]
        edges.append(np.unique(np.quantile(v, np.linspace(0,1,33)[1:-1])) if len(v) else None)
    return edges
def apply_bins(X, edges):
    Bx = np.zeros(X.shape, dtype=np.uint8)
    for f in range(X.shape[1]):
        qs = edges[f]
        if qs is None: continue
        col = X[:,f]
        b = np.searchsorted(qs, col, side='right')
        b[~np.isfinite(col)] = len(qs)+1
        Bx[:,f] = b.astype(np.uint8)
    return Bx

def gbm_df(df, trees=220, lr=0.08, depth=3, lam=1.0, mc=20, colsfrac=1.0, tag='', seed=0):
    dd = tt.merge(df, on=['household_key','snapshot_day'], how='inner')
    y = dd['future_spend_4w'].values.astype(float); days = dd['snapshot_day'].values
    base = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    X = dd[base].values.astype(float)
    tr = days < 431; te = days == 431
    edges = make_edges(X[tr]); Btr = apply_bins(X[tr], edges); Bte = apply_bins(X[te], edges)
    b0 = y[tr].mean(); p = np.full(tr.sum(), b0); pt = np.full(te.sum(), b0)
    rng = np.random.RandomState(seed); nf = X.shape[1]
    for t in range(trees):
        gr = p - y[tr]
        feats = rng.choice(nf, max(1,int(colsfrac*nf)), replace=False) if colsfrac<1 else np.arange(nf)
        nodes = {0:{}}; queue=[(np.arange(tr.sum()),0,0)]; i=0
        while i < len(queue):
            idx, dep, nid = queue[i]; i+=1; m=len(idx)
            if dep>=depth or m<2*mc:
                nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            sub = Btr[idx]; G = gr[idx]; best=(0.0,-1,-1)
            for f in feats:
                bh = np.bincount(sub[:,f], minlength=NB).astype(np.float64)
                bg = np.bincount(sub[:,f], weights=G, minlength=NB)
                ch = np.cumsum(bh); cg = np.cumsum(bg)
                GL,HL = cg[:-1],ch[:-1]; GR,HR = cg[-1]-GL, ch[-1]-HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - (GL+GR)**2/(HL+HR+lam)
                kk = int(np.argmax(gain))
                if gain[kk]>best[0] and ch[kk]>=mc and (ch[-1]-ch[kk])>=mc: best=(float(gain[kk]),f,kk)
            if best[1]<0:
                nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            gv,f,thr = best
            mask = sub[:,f]<=thr; L,R = idx[mask], idx[~mask]
            nodes[nid]['split']=(f,thr); lid=len(nodes); nodes[lid]={}; rid=len(nodes); nodes[rid]={}
            nodes[nid]['left'],nodes[nid]['right']=lid,rid
            queue.append((L,dep+1,lid)); queue.append((R,dep+1,rid))
        leafmap = {n_:nd['leaf'] for n_,nd in nodes.items() if 'leaf' in nd}
        cur = np.zeros(tr.sum(),dtype=np.int32); curt = np.zeros(te.sum(),dtype=np.int32)
        for nid in sorted(nodes):
            nd = nodes[nid]
            if 'split' in nd:
                f,thr = nd['split']
                s_ = cur==nid;  cur[s_ & (Btr[:,f]<=thr)]  = nd['left']; cur[s_ & (Btr[:,f]>thr)]  = nd['right']
                s_ = curt==nid; curt[s_ & (Bte[:,f]<=thr)] = nd['left']; curt[s_ & (Bte[:,f]>thr)] = nd['right']
        p  += lr*np.array([leafmap[c] for c in cur])
        pt += lr*np.array([leafmap.get(c,0.0) for c in curt])
    mae = float(np.mean(np.abs(pt-y[te])))
    print(f'GBM[{tag}] nf={nf} val431 MAE {mae:.3f}')
    return mae

def with_extra(df, extra):
    return df.merge(extra, left_index=True, right_index=True, how='left')

t0=time.time()
gbm_df(e006, colsfrac=0.5, tag='E006 cf.5')
gbm_df(e011, colsfrac=0.5, tag='E011 cf.5')
f011 = build_fine(e011)
for subset, name in [
    (['spend_7d','spend_14d','spend_21d'],'windows714'),
    (['trips_7d','trips_14d','ew_trips_hl28'],'tripsrec'),
    (['spend_7d','spend_14d','spend_21d','trips_7d','trips_14d','ew_trips_hl28','r7_28','r14_28','last_basket','phase'],'fine-all'),
]:
    gbm_df(with_extra(e011, f011[subset]), colsfrac=0.5, tag=f'E011+{name}')
f006 = build_fine(e006)
gbm_df(with_extra(e006, f006), colsfrac=0.5, tag='E006+fine-all')
print(f'{time.time()-t0:.0f}s')


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
d = tt.merge(e011, on=['household_key','snapshot_day'], how='inner')
y = d['future_spend_4w'].values.astype(float); days = d['snapshot_day'].values
allc = [c for c in e011.columns if c not in ('household_key','snapshot_day')]
X = d[allc].values.astype(float)

def ridge_variants(X, y, days, val_day=431):
    tr = days<val_day; te = days==val_day
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[~np.isfinite(sd)|(sd==0)]=1
    A = np.where(np.isnan(X[tr]),0,(X[tr]-mu)/sd); B = np.where(np.isnan(X[te]),0,(X[te]-mu)/sd)
    out={}
    for lam in [3,30,300]:
        w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@y[tr])
        out[f'ridge raw lam{lam}'] = float(np.mean(np.abs(B@w - y[te])))
    ly = np.log1p(y)
    for lam in [3,30,300]:
        w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@ly[tr])
        out[f'ridge log1p lam{lam}'] = float(np.mean(np.abs(np.expm1(B@w) - y[te])))
    # robust: huber via IRLS on raw
    w = np.zeros(X.shape[1]); 
    for it in range(15):
        r = y[tr]-A@w; s = np.median(np.abs(r-np.median(r)))*4+1e-9
        wt = np.minimum(1.0, s/np.maximum(np.abs(r),1e-9))
        W = A*(wt[:,None]); w = np.linalg.solve(W.T@A+30*np.eye(X.shape[1]), W.T@y[tr])
    out['huber raw'] = float(np.mean(np.abs(B@w - y[te])))
    ly = np.log1p(y)
    w = np.zeros(X.shape[1])
    for it in range(15):
        r = ly[tr]-A@w; s = np.median(np.abs(r-np.median(r)))*4+1e-9
        wt = np.minimum(1.0, s/np.maximum(np.abs(r),1e-9))
        W = A*(wt[:,None]); w = np.linalg.solve(W.T@A+30*np.eye(X.shape[1]), W.T@ly[tr])
    out['huber log1p'] = float(np.mean(np.abs(np.expm1(B@w) - y[te])))
    # knn on standardized features (cosine/cityblock)
    from collections import Counter
    best=[]
    for k in [15,40]:
        # subsample train for speed
        idx_tr = np.where(tr)[0]
        D = np.abs(A[idx_tr][:,None,:]-B[None,:,:]).sum(-1) if False else None
        # chunked L1
        n_te = te.sum(); preds=np.zeros(n_te)
        CH=500
        for c0 in range(0,n_te,CH):
            sl = slice(c0,min(c0+CH,n_te))
            dist = np.abs(A[idx_tr][:,None,:]-B[sl][None,:,:]).sum(-1)
            nn = np.argpartition(dist, k, axis=0)[:k]
            preds[sl] = y[tr][nn].mean(0)
        out[f'knn L1 k={k}'] = float(np.mean(np.abs(preds-y[te])))
    return out

t0=time.time()
for k,v in ridge_variants(X,y,days).items(): print(f'{k:24s} {v:8.3f}')
print(f'{time.time()-t0:.0f}s')


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
d = tt.merge(e011, on=['household_key','snapshot_day'], how='inner')
y = d['future_spend_4w'].values.astype(float); days = d['snapshot_day'].values
allc = [c for c in e011.columns if c not in ('household_key','snapshot_day')]
X = d[allc].values.astype(float)

def ridge_variants(X, y, days, val_day=431):
    tr = days<val_day; te = days==val_day
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[~np.isfinite(sd)|(sd==0)]=1
    A = np.where(np.isnan(X[tr]),0,(X[tr]-mu)/sd); B = np.where(np.isnan(X[te]),0,(X[te]-mu)/sd)
    out={}
    for lam in [3,30,300]:
        w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@y[tr])
        out[f'ridge raw lam{lam}'] = float(np.mean(np.abs(B@w - y[te])))
    ly = np.log1p(y)
    for lam in [3,30,300]:
        w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@ly[tr])
        out[f'ridge log1p lam{lam}'] = float(np.mean(np.abs(np.expm1(B@w) - y[te])))
    w = np.zeros(X.shape[1])
    for it in range(15):
        r = y[tr]-A@w; s = np.median(np.abs(r-nnp.median(r)))*4+1e-9 if False else np.median(np.abs(r-np.median(r)))*4+1e-9
        wt = np.minimum(1.0, s/np.maximum(np.abs(r),1e-9))
        W = A*(wt[:,None]); w = np.linalg.solve(W.T@A+30*np.eye(X.shape[1]), W.T@y[tr])
    out['huber raw'] = float(np.mean(np.abs(B@w - y[te])))
    w = np.zeros(X.shape[1])
    for it in range(15):
        r = ly[tr]-A@w; s = np.median(np.abs(r-np.median(r)))*4+1e-9
        wt = np.minimum(1.0, s/np.maximum(np.abs(r),1e-9))
        W = A*(wt[:,None]); w = np.linalg.solve(W.T@A+30*np.eye(X.shape[1]), W.T@ly[tr])
    out['huber log1p'] = float(np.mean(np.abs(np.expm1(B@w) - y[te])))
    for k in [15,40]:
        idx_tr = np.where(tr)[0]
        n_te = int(te.sum()); preds=np.zeros(n_te)
        CH=400
        for c0 in range(0,n_te,CH):
            sl = slice(c0,min(c0+CH,n_te))
            dist = np.abs(A[idx_tr][:,None,:]-B[sl][None,:,:]).sum(-1)
            nn = np.argpartition(dist, k, axis=0)[:k]
            preds[sl] = y[tr][nn].mean(0)
        out[f'knn L1 k={k}'] = float(np.mean(np.abs(preds-y[te])))
    return out

t0=time.time()
for k,v in ridge_variants(X,y,days).items(): print(f'{k:24s} {v:8.3f}')
print(f'{time.time()-t0:.0f}s')


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
d = tt.merge(e011, on=['household_key','snapshot_day'], how='inner')
y = d['future_spend_4w'].values.astype(float); days = d['snapshot_day'].values
allc = [c for c in e011.columns if c not in ('household_key','snapshot_day')]
X = d[allc].values.astype(float)

def variants(X, y, days, val_day=431):
    tr = days<val_day; te = days==val_day
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[~np.isfinite(sd)|(sd==0)]=1
    Z = np.where(np.isnan(X),0,(X-mu)/sd)
    A, B = Z[tr], Z[te]
    out={}
    ly = np.log1p(y)
    for lam in [3,30,300]:
        w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@y[tr])
        out[f'ridge raw lam{lam}'] = float(np.mean(np.abs(B@w - y[te])))
        w2 = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@ly[tr])
        out[f'ridge log1p lam{lam}'] = float(np.mean(np.abs(np.expm1(B@w2) - y[te])))
    # knn (indices into full Z)
    idx_tr = np.where(tr)[0]
    for k in [15,40]:
        n_te = int(te.sum()); preds=np.zeros(n_te); CH=400
        for c0 in range(0,n_te,CH):
            sl = slice(c0,min(c0+CH,n_te))
            dist = np.abs(Z[idx_tr][:,None,:]-Z[te][sl][None,:,:]).sum(-1)
            nn = np.argpartition(dist, k, axis=0)[:k]
            preds[sl] = y[tr][nn].mean(0)
        out[f'knn L1 k={k}'] = float(np.mean(np.abs(preds-y[te])))
    return out

t0=time.time()
res = variants(X,y,days)
for k,v in res.items(): print(f'{k:24s} {v:8.3f}')
print(f'{time.time()-t0:.0f}s')


# ---- cell ----
import pandas as pd, numpy as np, time
import agent_api
t0=time.time()
e011 = agent_api.load_saved('e011_table.parquet')
tt = agent_api.train_targets()
d = tt.merge(e011, on=['household_key','snapshot_day'], how='inner')
y = d['future_spend_4w'].values.astype(float); days = d['snapshot_day'].values
allc = [c for c in e011.columns if c not in ('household_key','snapshot_day')]
X = d[allc].values.astype(float)

def ridge_fit(X, y, days, val_day=431, lam=30, logt=False):
    tr = days<val_day; te = days==val_day
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[~np.isfinite(sd)|(sd==0)]=1
    Z = np.where(np.isnan(X),0,(X-mu)/sd)
    A,B = Z[tr],Z[te]; t = np.log1p(y) if logt else y
    w = np.linalg.solve(A.T@A+lam*np.eye(X.shape[1]), A.T@t[tr])
    p = B@w
    if logt: p = np.expm1(p)
    return float(np.mean(np.abs(p-y[te])))

print('log-ridge E011 lam3/30/300:', [round(ridge_fit(X,y,days,lam=l,logt=True),2) for l in (3,30,300)])
print('raw-ridge E011 lam30:', round(ridge_fit(X,y,days),2))

# ---- untried level-shifters: store, unit price, pack size, product breadth ----
snap = agent_api.snapshot(459)
tx = snap.transactions
prod = snap.products
sz = pd.to_numeric(prod['curr_size_of_product'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
pmap_sz = pd.Series(sz.values, index=prod['product_id'].values)
tx2 = tx[['household_key','day','sales_value','quantity','store_id','product_id','basket_id']].copy()
tx2['up'] = tx2['sales_value']/tx2['quantity'].replace(0,np.nan)
tx2['psz'] = tx2['product_id'].map(pmap_sz)

def build_lvl(df):
    hh = df['household_key'].values; S = df['snapshot_day'].values
    F = pd.DataFrame(index=df.index)
    sub = tx2[tx2['day']<=459]
    for k,s in zip(hh,S):
        pass
    # vectorized per household via groupby on precomputed slices
    gg = sub.groupby('household_key')
    out = {c:[] for c in ['top_store','store_share','unit_price_84','unit_price_364','psz_mean_364','nprod_364','nbask_364','store_id_num']}
    for k,s in zip(hh,S):
        try: idx = gg.indices[k]
        except KeyError: idx = np.array([],dtype=int)
        if len(idx)==0:
            for c in out: out[c].append(np.nan)
            continue
        dd = sub['day'].values[idx]; st = sub['store_id'].values[idx]
        sv = sub['sales_value'].values[idx]; up = sub['up'].values[idx]; pz = sub['psz'].values[idx]
        pid = sub['product_id'].values[idx]; bk = sub['basket_id'].values[idx]
        m364 = dd > s-364; m84 = dd > s-84
        if m364.sum()==0:
            for c in out: out[c].append(np.nan)
            continue
        svv = sv[m364]
        cs = pd.Series(svv).groupby(st[m364]).sum()
        out['top_store'].append(float(cs.idxmax())); out['store_id_num'].append(float(cs.idxmax()))
        out['store_share'].append(float(cs.max()/cs.sum()))
        u84 = up[m84]; u84 = u84[np.isfinite(u84)]
        out['unit_price_84'].append(float(np.median(u84)) if len(u84) else np.nan)
        u364 = up[m364]; u364 = u364[np.isfinite(u364)]
        out['unit_price_364'].append(float(np.median(u364)) if len(u364) else np.nan)
        pzv = pz[m364]; pzv = pzv[np.isfinite(pzv)]
        out['psz_mean_364'].append(float(np.median(pzv)) if len(pzv) else np.nan)
        out['nprod_364'].append(float(len(np.unique(pid[m364]))))
        out['nbask_364'].append(float(len(np.unique(bk[m364]))))
    return pd.DataFrame(out, index=df.index).astype(float)

L = build_lvl(d)
print('lvl built', round(time.time()-t0,1),'s')
# one-hot top stores (top 12 by frequency)
ts_counts = L['top_store'].value_counts()
top_stores = list(ts_counts.index[:12])
for s_ in top_stores: L[f'store_{s_}'] = (L['top_store']==s_).astype(float)
L['store_other'] = (~L['top_store'].isin(top_stores)).astype(float)
L = L.drop(columns=['top_store'])
lcols = list(L.columns)

def gbm_df(df, extra, trees=200, lr=0.08, depth=3, lam=1.0, mc=20, colsfrac=0.5, tag=''):
    dd = tt.merge(df, on=['household_key','snapshot_day'], how='inner')
    yv = dd['future_spend_4w'].values.astype(float); dv = dd['snapshot_day'].values
    base = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    Xv = dd[base].values.astype(float)
    if extra is not None:
        Xv = np.column_stack([Xv, extra.values.astype(float)])
    tr = dv<431; te = dv==431
    NB=34
    edges=[]
    for f in range(Xv.shape[1]):
        v = Xv[tr][:,f]; v = v[np.isfinite(v)]
        edges.append(np.unique(np.quantile(v, np.linspace(0,1,33)[1:-1])) if len(v) else None)
    def ab(Xa):
        Bx = np.zeros(Xa.shape, dtype=np.uint8)
        for f in range(Xa.shape[1]):
            qs = edges[f]
            if qs is None: continue
            col = Xa[:,f]; b = np.searchsorted(qs, col, side='right'); b[~np.isfinite(col)] = len(qs)+1
            Bx[:,f] = b.astype(np.uint8)
        return Bx
    Btr, Bte = ab(Xv[tr]), ab(Xv[te])
    b0 = yv[tr].mean(); p = np.full(tr.sum(), b0); pt = np.full(te.sum(), b0)
    rng = np.random.RandomState(0); nf = Xv.shape[1]
    for t in range(trees):
        gr = p - yv[tr]
        feats = rng.choice(nf, max(1,int(colsfrac*nf)), replace=False) if colsfrac<1 else np.arange(nf)
        nodes = {0:{}}; queue=[(np.arange(tr.sum()),0,0)]; i=0
        while i < len(queue):
            idx, dep, nid = queue[i]; i+=1; m=len(idx)
            if dep>=depth or m<2*mc: nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            sb = Btr[idx]; G = gr[idx]; best=(0.0,-1,-1)
            for f in feats:
                bh = np.bincount(sb[:,f], minlength=NB).astype(np.float64)
                bg = np.bincount(sb[:,f], weights=G, minlength=NB)
                ch = np.cumsum(bh); cg = np.cumsum(bg)
                GL,HL = cg[:-1],ch[:-1]; GR,HR = cg[-1]-GL, ch[-1]-HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - (GL+GR)**2/(HL+HR+lam)
                kk = int(np.argmax(gain))
                if gain[kk]>best[0] and ch[kk]>=mc and (ch[-1]-ch[kk])>=mc: best=(float(gain[kk]),f,kk)
            if best[1]<0: nodes[nid]['leaf'] = -gr[idx].sum()/(m+lam); continue
            gv,f,thr = best
            mask = sb[:,f]<=thr; Lq,Rq = idx[mask], idx[~mask]
            nodes[nid]['split']=(f,thr); lid=len(nodes); nodes[lid]={}; rid=len(nodes); nodes[rid]={}
            nodes[nid]['left'],nodes[nid]['right']=lid,rid
            queue.append((Lq,dep+1,lid)); queue.append((Rq,dep+1,rid))
        leafmap = {n_:nd['leaf'] for n_,nd in nodes.items() if 'leaf' in nd}
        cur = np.zeros(tr.sum(),dtype=np.int32); curt = np.zeros(te.sum(),dtype=np.int32)
        for nid in sorted(nodes):
            nd = nodes[nid]
            if 'split' in nd:
                f,thr = nd['split']
                s_ = cur==nid;  cur[s_ & (Btr[:,f]<=thr)]  = nd['left']; cur[s_ & (Btr[:,f]>thr)]  = nd['right']
                s_ = curt==nid; curt[s_ & (Bte[:,f]<=thr)] = nd['left']; curt[s_ & (Bte[:,f]>thr)] = nd['right']
        p  += lr*np.array([leafmap[c] for c in cur]); pt += lr*np.array([leafmap.get(c,0.0) for c in curt])
    mae = float(np.mean(np.abs(pt-yv[te])))
    print(f'GBM[{tag}] nf={nf} val431 MAE {mae:.3f}')
    return mae

gbm_df(e011, None, tag='E011 base cf.5')
gbm_df(e011, L[lcols], tag='E011+store/uprice/psz')
# log-ridge with log features added
logfeats = np.log1p(d[['spend_28d','spend_84d','spend_364d','ew_spend_hl28','usual_4w','sc_f_mean']].values.astype(float))
Xlog = np.column_stack([X, logfeats])
print('log-ridge E011+logfeats lam30:', round(ridge_fit(Xlog,y,days,logt=True),2))
print(f'{time.time()-t0:.0f}s')


# ---- cell ----
import pandas as pd, numpy as np
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
d = e011.copy()
snap = agent_api.snapshot(459)
tx = snap.transactions
prod = snap.products
sz = pd.to_numeric(prod['curr_size_of_product'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
pmap_sz = pd.Series(sz.values, index=prod['product_id'].values)
tx2 = tx[['household_key','day','sales_value','quantity','store_id','product_id','basket_id']].copy()
tx2['up'] = tx2['sales_value']/tx2['quantity'].replace(0,np.nan)
tx2['psz'] = tx2['product_id'].map(pmap_sz)
gg = tx2.groupby('household_key')
hh = d['household_key'].values; S = d['snapshot_day'].values
cols = ['store_share','unit_price_84','unit_price_364','psz_med_364','nprod_364','nbask_364','top_store']
out = {c: [] for c in cols}
for k, s in zip(hh, S):
    idx = gg.indices.get(k)
    if idx is None or len(idx)==0:
        for c in cols: out[c].append(np.nan)
        continue
    dd = tx2['day'].values[idx]; st = tx2['store_id'].values[idx]
    sv = tx2['sales_value'].values[idx]; up = tx2['up'].values[idx]; pz = tx2['psz'].values[idx]
    pid = tx2['product_id'].values[idx]; bk = tx2['basket_id'].values[idx]
    m364 = (dd > s-364) & (dd <= s); m84 = (dd > s-84) & (dd <= s)
    if m364.sum()==0:
        for c in cols: out[c].append(np.nan)
        continue
    cs = pd.Series(sv[m364]).groupby(st[m364]).sum()
    out['top_store'].append(float(cs.idxmax()))
    out['store_share'].append(float(cs.max()/cs.sum()))
    u84 = up[m84]; u84 = u84[np.isfinite(u84)]
    out['unit_price_84'].append(float(np.median(u84)) if len(u84) else np.nan)
    u364 = up[m364]; u364 = u364[np.isfinite(u364)]
    out['unit_price_364'].append(float(np.median(u364)) if len(u364) else np.nan)
    pzv = pz[m364]; pzv = pzv[np.isfinite(pzv)]
    out['psz_med_364'].append(float(np.median(pzv)) if len(pzv) else np.nan)
    out['nprod_364'].append(float(len(np.unique(pid[m364]))))
    out['nbask_364'].append(float(len(np.unique(bk[m364]))))
L = pd.DataFrame(out, index=d.index).astype(float)
ts_counts = L['top_store'].value_counts()
top_stores = list(ts_counts.index[:12])
for s_ in top_stores: L[f'store_{int(s_)}'] = (L['top_store']==s_).astype(float)
L['store_other'] = (~L['top_store'].isin(top_stores)).astype(float)
L = L.drop(columns=['top_store'])
L.insert(0,'household_key', d['household_key'].values)
L.insert(1,'snapshot_day', d['snapshot_day'].values)
print('lvl table', L.shape, list(L.columns)[:10])
path = agent_api.save_table(L, 'lvl_v1')
print('PATH:', path)
