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
