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
