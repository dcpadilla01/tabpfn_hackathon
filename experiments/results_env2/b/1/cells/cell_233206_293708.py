import agent_api, numpy as np, pandas as pd, time
t0=time.time()
tt = agent_api.train_targets()

def prep(path):
    t = agent_api.load_saved(path)
    feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    m = t.merge(tt, on=['household_key','snapshot_day'], how='left')
    isval = m.future_spend_4w.isna().values
    Xdf = m[feat].copy()
    for c in feat:
        if str(Xdf[c].dtype) in ('object','category','bool'):
            Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            Xdf[c] = Xdf[c].astype(float)
    return feat, Xdf.values, isval

def binize(Xref, Xnew, nbins=32):
    p = Xref.shape[1]
    edges = []; Bref = np.zeros(Xref.shape, np.int16); Bnew = np.zeros(Xnew.shape, np.int16)
    for j in range(p):
        col = Xref[:,j]; med = np.nanmedian(col)
        if np.isnan(med): med = 0.0
        colf = np.where(np.isnan(col), med, col)
        qs = np.unique(np.quantile(colf, np.linspace(0,1,nbins+1)[1:-1]))
        edges.append(qs)
        Bref[:,j] = np.searchsorted(qs, colf, 'right')
        coln = Xnew[:,j]; coln = np.where(np.isnan(coln), med, coln)
        Bnew[:,j] = np.searchsorted(qs, coln, 'right')
    return Bref, Bnew, edges

def gbm_fit(Btr, ytr, edges, depth=4, T=100, lr=0.1, lam=1.0, minleaf=20, seed=0, sub=0.85):
    rng = np.random.RandomState(seed); n, p = Btr.shape
    nb = np.array([len(e) for e in edges]); off = np.concatenate([[0], np.cumsum(nb+1)[:-1]])
    total = int(off[-1] + nb[-1] + 1); F = np.zeros(n); trees = []
    for it in range(T):
        resid = ytr - F; rows0 = rng.choice(n, int(n*sub), replace=False)
        leafval = {}; splits = {}; queue = [(rows0, 0, 0)]
        while queue:
            rows, d, nid = queue.pop()
            g = resid[rows]; G = g.sum(); H = len(rows)
            if d >= depth or H < 2*minleaf:
                leafval[nid] = lr*(G/(H+lam)); F[rows] += leafval[nid]; continue
            subM = Btr[rows]; flat = (subM.astype(np.int64)+off[None,:]).ravel()
            hist = np.bincount(flat, weights=np.repeat(g,p), minlength=total)
            cnt = np.bincount(flat, minlength=total)
            bg, bj, bb = 0.0, -1, -1
            for j in range(p):
                h = hist[off[j]:off[j]+nb[j]+1]; c = cnt[off[j]:off[j]+nb[j]+1]
                if len(h) < 2: continue
                GL = np.cumsum(h)[:-1]; HL = np.cumsum(c)[:-1]; GR = G-GL; HR = H-HL
                gain = GL*GL/(HL+lam)+GR*GR/(HR+lam)-G*G/(H+lam)
                gain[(HL<minleaf)|(HR<minleaf)] = -1
                b = int(np.argmax(gain))
                if gain[b] > bg: bg, bj, bb = gain[b], j, b
            if bj < 0:
                leafval[nid] = lr*(G/(H+lam)); F[rows] += leafval[nid]; continue
            mask = subM[:, bj] <= bb; lid, rid = nid*2+1, nid*2+2
            splits[nid] = (bj, bb, lid, rid)
            queue.append((rows[mask], d+1, lid)); queue.append((rows[~mask], d+1, rid))
        trees.append((splits, leafval))
    return trees

def gbm_predict(Bev, trees):
    out = np.zeros(Bev.shape[0])
    for splits, leafval in trees:
        active = {0: np.arange(Bev.shape[0])}
        while active:
            nid, rows = active.popitem()
            if nid in leafval: out[rows] += leafval[nid]; continue
            j, b, lid, rid = splits[nid]
            msk = Bev[rows, j] <= b
            if msk.any(): active[lid] = np.concatenate([active.get(lid, np.array([],int)), rows[msk]])
            if (~msk).any(): active[rid] = np.concatenate([active.get(rid, np.array([],int)), rows[~msk]])
    return out

def cv(path, tag):
    feat, X, isval = prep(path)
    sd = None
    t = agent_api.load_saved(path); sdv = t.snapshot_day.values.astype(int)
    y = None
    m = t.merge(tt, on=['household_key','snapshot_day'], how='left')
    yv = m.future_spend_4w.values
    maes = []
    for tr_max, evs in [(347,[375]), (375,[403,431])]:
        tr = sdv <= tr_max; ev = np.isin(sdv, evs)
        Btr, Bev, edges = binize(X[tr], X[ev])
        trees = gbm_fit(Btr, yv[tr].astype(float), edges)
        pr = gbm_predict(Bev, trees)
        maes.append(np.abs(pr - yv[ev].astype(float)).mean())
    print(f'{tag:14s} MAE {maes[0]:.2f} / {maes[1]:.2f}  AVG {np.mean(maes):.3f}  (p={X.shape[1]})')
    return np.mean(maes)

base = cv('e008_level_shape.parquet', 'E008')
for name, tag in [('candA_dm','+deptMom'),('candB_g','+global'),('candC_hb','+habit'),('candD_su','+stockup'),('candE_td','+timeofday')]:
    cv(name+'.parquet', tag)
# all combined
e8 = agent_api.load_saved('e008_level_shape.parquet')
dfall = e8
for name in ['candA_dm','candB_g','candC_hb','candD_su','candE_td']:
    t = agent_api.load_saved(name+'.parquet')
    newc = [c for c in t.columns if c not in dfall.columns]
    dfall = dfall.merge(t[['household_key','snapshot_day']+newc], on=['household_key','snapshot_day'], how='left')
p = agent_api.save_table(dfall, 'candAll')
cv('candAll.parquet', '+ALL')
print('done', round(time.time()-t0), 's')
