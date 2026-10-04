import agent_api, pandas as pd, numpy as np, datetime

t = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xdf = df[feat_cols]
M = np.empty((len(df), len(feat_cols)), dtype=np.float64)
for j,c in enumerate(feat_cols):
    col = Xdf[c].values.astype(np.float64)
    col[~np.isfinite(col)] = np.nan
    M[:,j] = col
train_mask = df['future_spend_4w'].notna().values
y = df['future_spend_4w'].values.astype(np.float64)
hh = df['household_key'].values
uq = np.unique(hh); rng = np.random.RandomState(7); rng.shuffle(uq)
hh_hold = set(uq[:int(len(uq)*0.3)])
ho = np.array([h in hh_hold for h in hh]) & train_mask
tr = train_mask & ~ho
med = np.nanmedian(M[tr], axis=0); med = np.where(np.isfinite(med), med, 0.0)
M = np.where(np.isnan(M), med[None,:], M)
q99 = np.quantile(np.abs(M[tr]), 0.999, axis=0)
M = np.clip(M, -q99[None,:], q99[None,:])

NB = 33
bins = np.empty((len(df), len(feat_cols)), dtype=np.uint8)
for j in range(len(feat_cols)):
    col = M[:,j]
    v = col[tr]; v = v[~np.isnan(v)]
    edges = np.unique(np.quantile(v, np.linspace(0,1,NB)[1:-1]))
    cc = np.where(np.isnan(col), np.inf, col)
    b = np.minimum(np.searchsorted(edges, cc, side='right'), NB-2)
    b[np.isnan(col)] = NB-1
    bins[:,j] = b.astype(np.uint8)

def build_tree(btr, grad, idx, fids, depth, min_leaf, rl, min_gain):
    nodes = []
    def add(sidx, d):
        G = grad[sidx].sum(); H = float(len(sidx))
        nid = len(nodes)
        nodes.append({'leaf':True,'val':-G/(H+rl),'f':-1,'thr':-1,'left':-1,'right':-1})
        if d >= depth or len(sidx) < 2*min_leaf:
            return nid
        bs = btr[np.ix_(sidx, fids)]
        gs = grad[sidx]
        best = None
        for kk in range(bs.shape[1]):
            b = bs[:,kk]
            hg = np.bincount(b, weights=gs, minlength=NB).cumsum()[:-1]
            hc = np.bincount(b, minlength=NB).cumsum()[:-1]
            nl = hc; nr = len(sidx)-hc
            ok = (nl>=min_leaf)&(nr>=min_leaf)
            if not ok.any(): continue
            GR = G-hg; HR = H-hc
            gain = hg*hg/(nl+rl) + GR*GR/(HR+rl) - G*G/(H+rl)
            gain[~ok] = -1e18
            jj = int(np.argmax(gain))
            if gain[jj] > min_gain and (best is None or gain[jj] > best[0]):
                best = (gain[jj], kk, jj)
        if best is None: return nid
        _, kk, thr = best
        mask = bs[:,kk] <= thr
        li = sidx[mask]; ri = sidx[~mask]
        if len(li)==0 or len(ri)==0: return nid
        nodes[nid].update({'leaf':False,'f':int(fids[kk]),'thr':int(thr)})
        nodes[nid]['left'] = add(li, d+1); nodes[nid]['right'] = add(ri, d+1)
        return nid
    root = add(idx, 0)
    return nodes, root

def tree_predict(nodes, brows):
    m = brows.shape[0]
    assign = np.zeros(m, dtype=np.int32)
    vals = np.array([nd['val'] for nd in nodes])
    for nid, nd in enumerate(nodes):
        if nd['leaf']: continue
        at = np.where(assign==nid)[0]
        if len(at)==0: continue
        gl = brows[at, nd['f']] <= nd['thr']
        assign[at[gl]] = nd['left']; assign[at[~gl]] = nd['right']
    return vals[assign]

def gbm_fit(btr, ytr, n_trees, lr=0.08, depth=4, min_leaf=60, rl=1.0, rs=0.8, cs=0.8, seed=0):
    rng = np.random.RandomState(seed)
    n, p = btr.shape
    base = ytr.mean()
    cur = np.full(n, base)
    trees = []
    for it in range(n_trees):
        grad = cur - ytr
        rows = np.where(rng.rand(n) < rs)[0]
        fids = np.where(rng.rand(p) < cs)[0]
        if len(fids) < 10: fids = np.arange(p)
        nodes, root = build_tree(btr, grad, rows, fids, depth, min_leaf, rl, 1e-6)
        upd = tree_predict(nodes, btr[rows])
        cur[rows] += lr*upd
        trees.append((nodes, root))
    return trees, base, lr

def gbm_predict(trees, base, lr, bidx):
    out = np.full(len(bidx), base)
    for nodes, root in trees:
        out += lr*tree_predict(nodes, bidx)
    return out

btr = bins[tr]; ytr_ = y[tr]
for (nt, lr, dep, ml) in [(150,0.08,4,60),(300,0.08,4,60),(300,0.08,6,60)]:
    t0 = datetime.datetime.now()
    trees, base, lr_ = gbm_fit(btr, ytr_, n_trees=nt, lr=lr, depth=dep, min_leaf=ml, seed=1)
    dt = (datetime.datetime.now()-t0).total_seconds()
    pv = gbm_predict(trees, base, lr_, bins[ho])
    pv2 = np.clip(pv, 0, None)
    print('nt=%d dep=%d: %.1fs holdout MAE %.3f (clip %.3f)' % (nt, dep, dt, np.abs(pv-y[ho]).mean(), np.abs(pv2-y[ho]).mean()))
