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
