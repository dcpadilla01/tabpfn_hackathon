import agent_api, numpy as np, pandas as pd, time

t = agent_api.load_saved('e008_level_shape.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'])
feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xdf = m[feat].copy()
for c in feat:
    if str(Xdf[c].dtype) in ('object','category','bool'):
        Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
    else:
        Xdf[c] = Xdf[c].astype(float)
X = Xdf.values; y = m.future_spend_4w.values.astype(float); sd = m.snapshot_day.values.astype(int)

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

tr = sd <= 347; ev = sd == 375
Btr, Bev, edges = binize(X[tr], X[ev])
nb = np.array([len(e) for e in edges]); off = np.concatenate([[0], np.cumsum(nb+1)[:-1]])
total = int(off[-1] + nb[-1] + 1)
print('nb sum', nb.sum(), 'total', total, 'max flat', (Btr.astype(int)+off[None,:]).max())

# tiny fit debug: T=3, depth=3, no subsample
def fit_debug(Btr, ytr, depth=3, T=3, lr=0.1, lam=1.0, minleaf=20):
    n, p = Btr.shape
    F = np.zeros(n); trees = []
    for it in range(T):
        resid = ytr - F
        print('iter', it, 'resid absmax', np.abs(resid).max())
        leafval = {}; splits = {}; queue = [(np.arange(n), 0, 0)]
        while queue:
            rows, d, nid = queue.pop()
            g = resid[rows]; G = g.sum(); H = len(rows)
            if d >= depth or H < 2*minleaf:
                leafval[nid] = lr * (-G/(H+lam)); F[rows] += leafval[nid]; continue
            subM = Btr[rows]
            flat = (subM.astype(np.int64) + off[None,:]).ravel()
            w = np.repeat(g, p)
            hist = np.bincount(flat, weights=w, minlength=total)
            cnt = np.bincount(flat, minlength=total)
            bg, bj, bb = 0.0, -1, -1
            for j in range(p):
                h = hist[off[j]:off[j]+nb[j]+1]; c = cnt[off[j]:off[j]+nb[j]+1]
                if len(h) < 2: continue
                GL = np.cumsum(h)[:-1]; HL = np.cumsum(c)[:-1]
                GR = G - GL; HR = H - HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - G*G/(H+lam)
                gain[(HL<minleaf)|(HR<minleaf)] = -1
                b = int(np.argmax(gain))
                if gain[b] > bg: bg, bj, bb = gain[b], j, b
            if bj < 0:
                leafval[nid] = lr * (-G/(H+lam)); F[rows] += leafval[nid]; continue
            mask = subM[:, bj] <= bb
            lid, rid = nid*2+1, nid*2+2
            splits[nid] = (bj, bb, lid, rid)
            queue.append((rows[mask], d+1, lid)); queue.append((rows[~mask], d+1, rid))
        trees.append((splits, leafval))
        print('  F absmax', np.abs(F).max(), 'y absmax', np.abs(ytr).max())
    return trees

trees = fit_debug(Btr, y[tr])
# predict check
def pred_tree(Bev, splits, leafval):
    out = np.zeros(Bev.shape[0])
    active = {0: np.arange(Bev.shape[0])}
    guard = 0
    while active:
        nid, rows = active.popitem()
        guard += 1
        if guard > 100000: print('GUARD HIT'); break
        if nid in leafval: out[rows] += leafval[nid]; continue
        j, b, lid, rid = splits[nid]
        msk = Bev[rows, j] <= b
        if msk.any(): active[lid] = np.concatenate([active.get(lid, np.array([],int)), rows[msk]])
        if (~msk).any(): active[rid] = np.concatenate([active.get(rid, np.array([],int)), rows[~msk]])
    return out
pr = np.zeros(Bev.shape[0])
for splits, leafval in trees:
    pr += pred_tree(Bev, splits, leafval)
print('pred stats', np.nanmin(pr), np.nanmax(pr), np.nanmean(pr))
print('MAE', np.abs(pr - y[ev]).mean())
