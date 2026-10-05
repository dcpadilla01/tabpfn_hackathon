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
