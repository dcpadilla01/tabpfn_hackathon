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

def build_tree(btr, grad, resid, idx, fids, depth, min_leaf, rl, l1_leaf):
    nodes = []
    def add(sidx, d):
        G = grad[sidx].sum(); H = float(len(sidx))
        nid = len(nodes)
        lv = float(np.median(resid[sidx])) if l1_leaf else -G/(H+rl)
        nodes.append({'leaf':True,'val':lv,'f':-1,'thr':-1,'left':-1,'right':-1})
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
    assign = np.zeros(brows.shape[0], dtype=np.int32)
    vals = np.array([nd['val'] for nd in nodes])
    for nid, nd in enumerate(nodes):
        if nd['leaf']: continue
        at = np.where(assign==nid)[0]
        if len(at)==0: continue
        gl = brows[at, nd['f']] <= nd['thr']
        assign[at[gl]] = nd['left']; assign[at[~gl]] = nd['right']
    return vals[assign]

def gbm_fit(btr, ytr, n_trees, lr, depth, min_leaf, rl, rs, cs, seed, l1_leaf, log_t):
    rng = np.random.RandomState(seed)
    n, p = btr.shape
    yt = np.log1p(ytr) if log_t else ytr
    base = float(np.median(yt)) if l1_leaf else float(yt.mean())
    cur = np.full(n, base)
    trees = []
    for it in range(n_trees):
        resid = yt - cur
        grad = -np.sign(resid) if l1_leaf else -resid
        rows = np.where(rng.rand(n) < rs)[0]
        fids = np.where(rng.rand(p) < cs)[0]
        if len(fids) < 10: fids = np.arange(p)
        nodes, root = build_tree(btr, grad, resid, rows, fids, depth, min_leaf, rl, l1_leaf)
        upd = tree_predict(nodes, btr[rows])
        cur[rows] += lr*upd
        trees.append((nodes, root))
    return trees, base

def gbm_predict(trees, base, lr, bidx):
    out = np.full(len(bidx), base)
    for nodes, root in trees:
        out += lr*tree_predict(nodes, bidx)
    return out

btr = bins[tr]; ytr_ = y[tr]
def run(name, nt, lr, dep, ml, rl, l1, logt, seed=1):
    t0 = datetime.datetime.now()
    trees, base = gbm_fit(btr, ytr_, nt, lr, dep, ml, rl, 0.8, 0.8, seed, l1, logt)
    dt = (datetime.datetime.now()-t0).total_seconds()
    pl = gbm_predict(trees, base, lr, bins[ho])
    if logt:
        p = np.expm1(pl)
    else:
        p = pl
    p = np.clip(p, 0, None)
    mae = np.abs(p - y[ho]).mean()
    # calibrate scalar on tr rows (fit on train only)
    ptr = gbm_predict(trees, base, lr, bins[tr])
    ptr = np.clip(np.expm1(ptr) if logt else ptr, 0, None)
    r = y[tr]/np.maximum(ptr, 1e-6)
    a = float(np.median(r))
    mae_c = np.abs(a*p - y[ho]).mean()
    print('%s: %.0fs MAE %.3f cal(a=%.3f) %.3f' % (name, dt, mae, a, mae_c))
    return mae

run('L2 log nt400 d4', 400, 0.05, 4, 100, 5.0, False, True)
run('L1 raw nt300 d4', 300, 0.10, 4, 60, 5.0, True, False)
run('L1 log nt300 d4', 300, 0.10, 4, 60, 5.0, True, True)
