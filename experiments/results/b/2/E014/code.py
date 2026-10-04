import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e013_denoise.parquet')
print('shape', t.shape)
print('cols:', list(t.columns))
tt = agent_api.train_targets()
print(tt['future_spend_4w'].describe().round(2))
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
num = [c for c in t.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
cor = m[num + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w')
print(cor.reindex(cor.abs().sort_values(ascending=False).index).head(45).round(3))

# ---- cell ----
import agent_api, pandas as pd, numpy as np, datetime

t = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xdf = df[feat_cols]
cols_num = [c for c in feat_cols if pd.api.types.is_numeric_dtype(Xdf[c])]
cols_cat = [c for c in feat_cols if c not in cols_num]
print('num', len(cols_num), 'cat', cols_cat)

M = np.empty((len(df), len(feat_cols)), dtype=np.float64)
for j,c in enumerate(feat_cols):
    if c in cols_num:
        M[:,j] = Xdf[c].values.astype(np.float64)
    else:
        M[:,j] = pd.Categorical(Xdf[c].astype(str)).codes

train_mask = df['future_spend_4w'].notna().values
y = df['future_spend_4w'].values.astype(np.float64)
tri = np.where(train_mask)[0]; vai = np.where(~train_mask)[0]
print('train', len(tri), 'val', len(vai))

hh = df['household_key'].values
uq = np.unique(hh); rng = np.random.RandomState(7); rng.shuffle(uq)
hh_hold = set(uq[:int(len(uq)*0.3)])
ho = np.array([h in hh_hold for h in hh]) & train_mask
tr = train_mask & ~ho
mu = M[tr].mean(0); sd = M[tr].std(0)+1e-9
Z = (M-mu)/sd
def ols(trm, vam):
    A = np.hstack([Z[trm], np.ones((trm.sum(),1))])
    coef,*_ = np.linalg.lstsq(A, y[trm], rcond=None)
    B = np.hstack([Z[vam], np.ones((vam.sum(),1))])
    return np.abs(B@coef - y[vam]).mean()
print('OLS holdout MAE %.3f' % ols(tr, ho))

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
print('bins done')

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
    return trees, base

def gbm_predict(trees, base, bidx):
    out = np.full(len(bidx), base)
    for nodes, root in trees:
        out += tree_predict(nodes, bidx)
    return out

btr = bins[tr]
ytr_ = y[tr]
t0 = datetime.datetime.now()
trees, base = gbm_fit(btr, ytr_, n_trees=10, seed=1)
dt = (datetime.datetime.now()-t0).total_seconds()
print('10 trees: %.2fs -> %.3fs/tree' % (dt, dt/10))
pv = gbm_predict(trees, base, bins[ho])
print('GBM10 holdout MAE %.3f' % np.abs(pv - y[ho]).mean())


# ---- cell ----
import agent_api, pandas as pd, numpy as np, datetime

t = agent_api.load_saved('e013_denoise.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xdf = df[feat_cols]
# check for inf / weird
bad = []
for c in feat_cols:
    v = Xdf[c]
    if pd.api.types.is_numeric_dtype(v):
        n_inf = np.isinf(v.values).sum(); n_nan = np.isnan(v.values).sum()
        mx = np.nanmax(np.abs(v.values)) if n_nan < len(v) else 0
        if n_inf or n_nan or mx > 1e12: bad.append((c, n_nan, n_inf, mx))
print('bad cols:', bad[:20], 'count', len(bad))

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
# fill nan with train median, clip huge
med = np.nanmedian(M[tr], axis=0)
med = np.where(np.isfinite(med), med, 0.0)
M = np.where(np.isnan(M), med[None,:], M)
q99 = np.quantile(np.abs(M[tr]), 0.999, axis=0)
M = np.clip(M, -q99[None,:], q99[None,:])
mu = M[tr].mean(0); sd = M[tr].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd
def ols(trm, vam):
    A = np.hstack([Z[trm], np.ones((trm.sum(),1))])
    coef,*_ = np.linalg.lstsq(A, y[trm], rcond=None)
    B = np.hstack([Z[vam], np.ones((vam.sum(),1))])
    return np.abs(B@coef - y[vam]).mean()
print('OLS holdout MAE %.3f' % ols(tr, ho))

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
print('bins done')

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
    return trees, base

def gbm_predict(trees, base, bidx):
    out = np.full(len(bidx), base)
    for nodes, root in trees:
        out += tree_predict(nodes, bidx)
    return out

btr = bins[tr]
ytr_ = y[tr]
t0 = datetime.datetime.now()
trees, base = gbm_fit(btr, ytr_, n_trees=10, seed=1)
dt = (datetime.datetime.now()-t0).total_seconds()
print('10 trees: %.2fs -> %.3fs/tree' % (dt, dt/10))
pv = gbm_predict(trees, base, bins[ho])
print('GBM10 holdout MAE %.3f' % np.abs(pv - y[ho]).mean())


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
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
            if gain[jj] > 1e-6 and (best is None or gain[jj] > best[0]):
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
    p = np.clip(np.expm1(pl) if logt else pl, 0, None)
    mae = np.abs(p - y[ho]).mean()
    ptr = np.clip(np.expm1(gbm_predict(trees, base, lr, bins[tr])) if logt else gbm_predict(trees, base, lr, bins[tr]), 0, None)
    a = float(np.median(y[tr]/np.maximum(ptr,1e-6)))
    mae_c = np.abs(a*p - y[ho]).mean()
    print('%s: %.0fs MAE %.3f cal(%.3f) %.3f' % (name, dt, mae, a, mae_c))

run('L2 log nt400 d4', 400, 0.05, 4, 100, 5.0, False, True)
run('L1 raw nt300 d4', 300, 0.10, 4, 60, 5.0, True, False)
run('L1 log nt300 d4', 300, 0.10, 4, 60, 5.0, True, True)


# ---- cell ----
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
mu = M[tr].mean(0); sd = M[tr].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd
A = np.hstack([Z[tr], np.ones((tr.sum(),1))])
coef,*_ = np.linalg.lstsq(A, y[tr], rcond=None)
B = np.hstack([Z, np.ones((len(df),1))])
ols_all = B@coef
print('OLS holdout MAE %.3f' % np.abs(ols_all[ho]-y[ho]).mean())

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
            if gain[jj] > 1e-6 and (best is None or gain[jj] > best[0]):
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

def boost_from_init(btr, resid_tr, init_all, n_trees, lr, depth, min_leaf, rl, rs, cs, seed, l1_leaf):
    rng = np.random.RandomState(seed)
    n, p = btr.shape
    cur = resid_tr.copy()
    trees = []
    for it in range(n_trees):
        grad = -np.sign(cur) if l1_leaf else -cur
        rows = np.where(rng.rand(n) < rs)[0]
        fids = np.where(rng.rand(p) < cs)[0]
        if len(fids) < 10: fids = np.arange(p)
        nodes, root = build_tree(btr, grad, cur, rows, fids, depth, min_leaf, rl, l1_leaf)
        upd = tree_predict(nodes, btr[rows])
        cur[rows] += lr*upd
        trees.append((nodes, root))
    # apply to all rows
    corr = np.zeros(len(init_all))
    for nodes, root in trees:
        corr += lr*tree_predict(nodes, bins)
    return init_all + corr

btr = bins[tr]
def run(name, nt, lr, dep, ml, rl, l1, rs, cs, seed=1):
    t0 = datetime.datetime.now()
    pred = boost_from_init(btr, y[tr]-ols_all[tr], ols_all, nt, lr, dep, ml, rl, rs, cs, seed, l1)
    dt = (datetime.datetime.now()-t0).total_seconds()
    p = np.clip(pred, 0, None)
    mae = np.abs(p[ho]-y[ho]).mean()
    a = float(np.median(y[tr]/np.maximum(pred[tr],1e-6)))
    mae_c = np.abs(a*p[ho]-y[ho]).mean()
    print('%s: %.0fs MAE %.3f cal(%.3f) %.3f' % (name, dt, mae, a, mae_c))

run('OLS+L1 nt300 lr.1 d4', 300, 0.10, 4, 60, 5.0, True, 0.8, 0.8)
run('OLS+L1 nt600 lr.05 d4', 600, 0.05, 4, 60, 5.0, True, 0.8, 0.8)
run('OLS+L2 nt600 lr.05 d4', 600, 0.05, 4, 60, 5.0, False, 0.8, 0.8)


# ---- cell ----
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
mu = M[tr].mean(0); sd = M[tr].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd
A = np.hstack([Z[tr], np.ones((tr.sum(),1))])
coef,*_ = np.linalg.lstsq(A, y[tr], rcond=None)
B = np.hstack([Z, np.ones((len(df),1))])
ols_all = B@coef
print('ols_all quantiles:', np.quantile(ols_all, [0,.01,.5,.99,1]).round(2))
resid_tr = y[tr]-ols_all[tr]
print('resid_tr quantiles:', np.quantile(resid_tr, [0,.01,.5,.99,1]).round(2))
print('y quantiles:', np.quantile(y[tr], [0,.5,.99,1]).round(2))


# ---- cell ----
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
mu = M[tr].mean(0); sd = M[tr].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd
A = np.hstack([Z[tr], np.ones((tr.sum(),1))])
coef,*_ = np.linalg.lstsq(A, y[tr], rcond=None)
B = np.hstack([Z, np.ones((len(df),1))])
ols_all = B@coef
print('OLS holdout MAE %.3f' % np.abs(ols_all[ho]-y[ho]).mean())

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
            if gain[jj] > 1e-6 and (best is None or gain[jj] > best[0]):
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

def boost_resid(btr, resid0, n_trees, lr, depth, min_leaf, rl, rs, cs, seed, l1_leaf, monitor=None):
    """cur = residual; update cur -= lr*upd; return list of trees"""
    rng = np.random.RandomState(seed)
    n, p = btr.shape
    cur = resid0.copy()
    trees = []
    for it in range(n_trees):
        grad = -np.sign(cur) if l1_leaf else -cur
        rows = np.where(rng.rand(n) < rs)[0]
        fids = np.where(rng.rand(p) < cs)[0]
        if len(fids) < 10: fids = np.arange(p)
        nodes, root = build_tree(btr, grad, cur, rows, fids, depth, min_leaf, rl, l1_leaf)
        upd = tree_predict(nodes, btr[rows])
        cur[rows] -= lr*upd
        trees.append((nodes, root))
        if monitor and (it+1) % monitor == 0:
            print('  tree %d: |cur| max %.1f med|%.1f|' % (it+1, np.abs(cur).max(), np.median(np.abs(cur))))
    return trees

def apply_trees(trees, lr, bidx):
    out = np.zeros(len(bidx))
    for nodes, root in trees:
        out += lr*tree_predict(nodes, bidx)
    return out

btr = bins[tr]
resid0 = y[tr] - ols_all[tr]
def run(name, nt, lr, dep, ml, rl, l1, seed=1):
    t0 = datetime.datetime.now()
    trees = boost_resid(btr, resid0, nt, lr, dep, ml, rl, 0.8, 0.8, seed, l1, monitor=100 if nt>400 else 0)
    dt = (datetime.datetime.now()-t0).total_seconds()
    corr_tr = apply_trees(trees, lr, bins[tr])
    corr_ho = apply_trees(trees, lr, bins[ho])
    p_ho = ols_all[ho] + corr_ho
    mae = np.abs(np.clip(p_ho,0,None) - y[ho]).mean()
    # calibrate on tr
    r = y[tr]/np.maximum(ols_all[tr]+corr_tr, 1e-6)
    a = float(np.median(r))
    mae_c = np.abs(np.clip(a*p_ho,0,None) - y[ho]).mean()
    print('%s: %.0fs MAE %.3f cal(%.3f) %.3f' % (name, dt, mae, a, mae_c))

run('L1 nt400 lr.05 d4 ml100', 400, 0.05, 4, 100, 5.0, True)
run('L1 nt400 lr.05 d5 ml50', 400, 0.05, 5, 50, 5.0, True)
run('L2 nt400 lr.05 d4 ml100', 400, 0.05, 4, 100, 5.0, False)


# ---- cell ----
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
mu = M[tr].mean(0); sd = M[tr].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd
A = np.hstack([Z[tr], np.ones((tr.sum(),1))])
coef,*_ = np.linalg.lstsq(A, y[tr], rcond=None)
B = np.hstack([Z, np.ones((len(df),1))])
ols_all = B@coef

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
            if gain[jj] > 1e-6 and (best is None or gain[jj] > best[0]):
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

def boost_resid(btr, resid0, n_trees, lr, depth, min_leaf, rl, rs, cs, seed, l1_leaf):
    rng = np.random.RandomState(seed)
    n, p = btr.shape
    cur = resid0.copy()
    trees = []
    for it in range(n_trees):
        grad = -np.sign(cur) if l1_leaf else -cur
        rows = np.where(rng.rand(n) < rs)[0]
        fids = np.where(rng.rand(p) < cs)[0]
        if len(fids) < 10: fids = np.arange(p)
        nodes, root = build_tree(btr, grad, cur, rows, fids, depth, min_leaf, rl, l1_leaf)
        upd = tree_predict(nodes, btr[rows])
        cur[rows] -= lr*upd
        trees.append((nodes, root))
    return trees

def apply_trees(trees, lr, bidx):
    out = np.zeros(len(bidx))
    for nodes, root in trees:
        out += lr*tree_predict(nodes, bidx)
    return out

btr = bins[tr]
resid0 = y[tr] - ols_all[tr]
def run(name, nt, lr, dep, ml, rl, rs, cs, seed=1):
    t0 = datetime.datetime.now()
    trees = boost_resid(btr, resid0, nt, lr, dep, ml, rl, rs, cs, seed, True)
    dt = (datetime.datetime.now()-t0).total_seconds()
    corr_tr = apply_trees(trees, lr, bins[tr])
    corr_ho = apply_trees(trees, lr, bins[ho])
    p_ho = ols_all[ho] + corr_ho
    mae = np.abs(np.clip(p_ho,0,None) - y[ho]).mean()
    print('%s: %.0fs MAE %.3f' % (name, dt, mae))

run('L1 nt800 lr.025 d4 ml100', 800, 0.025, 4, 100, 5.0, 0.8, 0.8)
run('L1 nt400 lr.05 d4 ml30', 400, 0.05, 4, 30, 5.0, 0.8, 0.8)
run('L1 nt600 lr.05 d4 ml100 rs1.0', 600, 0.05, 4, 100, 5.0, 1.0, 0.8)
run('L1 nt600 lr.05 d4 ml100 cs0.5', 600, 0.05, 4, 100, 5.0, 0.8, 0.5)

# ---- cell ----
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
mu = M[tr].mean(0); sd = M[tr].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd

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

def build_tree(btr, grad, resid, idx, fids, depth, min_leaf, rl):
    nodes = []
    def add(sidx, d):
        G = grad[sidx].sum(); H = float(len(sidx))
        nid = len(nodes)
        nodes.append({'leaf':True,'val':float(np.median(resid[sidx])),'f':-1,'thr':-1,'left':-1,'right':-1})
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
            if gain[jj] > 1e-6 and (best is None or gain[jj] > best[0]):
                best = (gain[jj], kk, jj)
        if best is None: return nid
        _, kk, thr = best
        mask = bs[:,kk] <= thr
        li = sidx[mask]; ri = sidx[~mask]
        if len(li)==0 or len(ri)==0: return nid
        nodes[nid].update({'leaf':False,'f':int(fids[kk]),'thr':int(thr)})
        nodes[nid]['left'] = add(li, d+1); nodes[nid]['right'] = add(ri, d+1)
        return nid
    return add(idx, 0), nodes

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

def boost_resid(btr, resid0, n_trees, lr, depth, min_leaf, rl, rs, cs, seed):
    rng = np.random.RandomState(seed)
    n, p = btr.shape
    cur = resid0.copy(); trees = []
    for it in range(n_trees):
        grad = -np.sign(cur)
        rows = np.where(rng.rand(n) < rs)[0]
        fids = np.where(rng.rand(p) < cs)[0]
        if len(fids) < 10: fids = np.arange(p)
        root, nodes = build_tree(btr, grad, cur, rows, fids, depth, min_leaf, rl)
        cur[rows] -= lr*tree_predict(nodes, btr[rows])
        trees.append(nodes)
    return trees

def apply_trees(trees, lr, bidx):
    out = np.zeros(len(bidx))
    for nodes in trees:
        out += lr*tree_predict(nodes, bidx)
    return out

btr = bins[tr]
t0 = datetime.datetime.now()

# (a) raw OLS + L1 boost, 2 seeds averaged
A = np.hstack([Z[tr], np.ones((tr.sum(),1))])
coef,*_ = np.linalg.lstsq(A, y[tr], rcond=None)
B = np.hstack([Z, np.ones((len(df),1))])
ols_all = B@coef
resid0 = y[tr] - ols_all[tr]
corr = np.zeros(len(df))
for s in (1,2):
    trees = boost_resid(btr, resid0, 500, 0.05, 4, 100, 5.0, 0.8, 0.5, s)
    corr += apply_trees(trees, 0.05, bins)
corr /= 2
p_a = np.clip(ols_all + corr, 0, None)
print('a) raw OLS+L1x2seed: %.0fs MAE %.3f' % ((datetime.datetime.now()-t0).total_seconds(), np.abs(p_a[ho]-y[ho]).mean()))

# (b) log OLS + L1 boost on log scale
t0 = datetime.datetime.now()
ly = np.log1p(y)
Al = np.hstack([Z[tr], np.ones((tr.sum(),1))])
coefl,*_ = np.linalg.lstsq(Al, ly[tr], rcond=None)
Bl = np.hstack([Z, np.ones((len(df),1))])
olsl_all = Bl@coefl
residl0 = ly[tr] - olsl_all[tr]
corrl = np.zeros(len(df))
for s in (1,2):
    trees = boost_resid(btr, residl0, 500, 0.05, 4, 100, 5.0, 0.8, 0.5, s)
    corrl += apply_trees(trees, 0.05, bins)
corrl /= 2
pl = np.expm1(olsl_all + corrl)
r = ly[tr] - (olsl_all[tr]+corrl[tr])
# median-ratio calibration on raw scale
ptr = np.clip(pl[tr], 0, None)
a = float(np.median(y[tr]/np.maximum(ptr,1e-6)))
p_b = np.clip(a*pl, 0, None)
print('b) log OLS+L1x2seed cal: %.0fs MAE %.3f (uncal %.3f)' % ((datetime.datetime.now()-t0).total_seconds(), np.abs(p_b[ho]-y[ho]).mean(), np.abs(np.clip(pl,0,None)[ho]-y[ho]).mean()))

# (c) blend a and b
for w in (0.5, 0.7):
    p_c = w*p_a + (1-w)*p_b
    print('c) blend w=%.1f: MAE %.3f' % (w, np.abs(p_c[ho]-y[ho]).mean()))


# ---- cell ----
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
mu = M[tr].mean(0); sd = M[tr].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd

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

def build_tree(btr, grad, resid, idx, fids, depth, min_leaf, rl):
    nodes = []
    def add(sidx, d):
        G = grad[sidx].sum(); H = float(len(sidx))
        nid = len(nodes)
        nodes.append({'leaf':True,'val':float(np.median(resid[sidx])),'f':-1,'thr':-1,'left':-1,'right':-1})
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
            if gain[jj] > 1e-6 and (best is None or gain[jj] > best[0]):
                best = (gain[jj], kk, jj)
        if best is None: return nid
        _, kk, thr = best
        mask = bs[:,kk] <= thr
        li = sidx[mask]; ri = sidx[~mask]
        if len(li)==0 or len(ri)==0: return nid
        nodes[nid].update({'leaf':False,'f':int(fids[kk]),'thr':int(thr)})
        nodes[nid]['left'] = add(li, d+1); nodes[nid]['right'] = add(ri, d+1)
        return nid
    return add(idx, 0), nodes

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

def boost_resid(btr, resid0, n_trees, lr, depth, min_leaf, rl, rs, cs, seed):
    rng = np.random.RandomState(seed)
    n, p = btr.shape
    cur = resid0.copy(); trees = []
    for it in range(n_trees):
        grad = -np.sign(cur)
        rows = np.where(rng.rand(n) < rs)[0]
        fids = np.where(rng.rand(p) < cs)[0]
        if len(fids) < 10: fids = np.arange(p)
        root, nodes = build_tree(btr, grad, cur, rows, fids, depth, min_leaf, rl)
        cur[rows] -= lr*tree_predict(nodes, btr[rows])
        trees.append(nodes)
    return trees

def apply_trees(trees, lr, bidx):
    out = np.zeros(len(bidx))
    for nodes in trees:
        out += lr*tree_predict(nodes, bidx)
    return out

def stack_fit(trm, nt=500, lr=0.05, dep=4, ml=100, rl=5.0, rs=0.8, cs=0.5):
    A = np.hstack([Z[trm], np.ones((trm.sum(),1))])
    coef,*_ = np.linalg.lstsq(A, y[trm], rcond=None)
    B = np.hstack([Z, np.ones((len(df),1))])
    ols_all = B@coef
    resid0 = y[trm] - ols_all[trm]
    btr = bins[trm]
    corr = np.zeros(len(df))
    for s in (1,2):
        trees = boost_resid(btr, resid0, nt, lr, dep, ml, rl, rs, cs, s)
        corr += apply_trees(trees, lr, bins)
    corr /= 2
    return ols_all, corr

t0 = datetime.datetime.now()
ols_tr, corr_tr70 = stack_fit(tr)
p_direct = np.clip(ols_tr + corr_tr70, 0, 2500)
print('direct stacked (holdout): MAE %.3f' % np.abs(p_direct[ho]-y[ho]).mean())

# simulate harness: refit OLS with gbm_pred as extra feature, train rows in-sample
gbm_is = ols_tr + corr_tr70
Xtr = np.hstack([Z[tr], gbm_is[tr][:,None]])
Xho = np.hstack([Z[ho], gbm_is[ho][:,None]])
c,*_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
print('SIM evaluator-OLS with gbm_pred feature: MAE %.3f  (%.0fs)' % (np.abs(Xho@c - y[ho]).mean(), (datetime.datetime.now()-t0).total_seconds()))

# FINAL: fit on ALL train rows
ols_all, corr_all = stack_fit(train_mask)
gbm_pred = np.clip(ols_all + corr_all, 0, 2500)
print('final gbm_pred quantiles:', np.quantile(gbm_pred, [0,.5,.9,.99,1]).round(1))
out = t.copy()
out['gbm_pred'] = gbm_pred
agent_api.save_table(out, 'e014_base.parquet')

# build via build_features (trivial attach fn); fallback to direct save
ok = False
try:
    def fn(view, snapshot_day):
        full = agent_api.load_saved('e014_base.parquet')
        sub = full[full['snapshot_day'] == snapshot_day].set_index('household_key')
        sub = sub.drop(columns=['snapshot_day'])
        return sub.reindex(pd.Index(list(view.households)))
    bf = agent_api.build_features(fn)
    print('build_features ok:', bf.shape)
    agent_api.save_table(bf, 'e014_stack.parquet')
    ok = True
except Exception as e:
    print('build_features FAILED:', repr(e)[:300])
if not ok:
    agent_api.save_table(out, 'e014_stack.parquet')
    print('fallback direct save done')
print('total %.0fs' % (datetime.datetime.now()-t0).total_seconds())


# ---- cell ----
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
uq = np.unique(hh); rng0 = np.random.RandomState(7); rng0.shuffle(uq)
hh_hold = set(uq[:int(len(uq)*0.3)])
ho = np.array([h in hh_hold for h in hh]) & train_mask
tr = train_mask & ~ho
med = np.nanmedian(M[train_mask], axis=0); med = np.where(np.isfinite(med), med, 0.0)
M = np.where(np.isnan(M), med[None,:], M)
q99 = np.quantile(np.abs(M[train_mask]), 0.999, axis=0)
M = np.clip(M, -q99[None,:], q99[None,:])
mu = M[train_mask].mean(0); sd = M[train_mask].std(0); sd[sd<1e-9]=1
Z = (M-mu)/sd

NB = 33
bins = np.empty((len(df), len(feat_cols)), dtype=np.uint8)
for j in range(len(feat_cols)):
    col = M[:,j]
    v = col[train_mask]
    edges = np.unique(np.quantile(v, np.linspace(0,1,NB)[1:-1]))
    cc = np.where(np.isnan(col), np.inf, col)
    b = np.minimum(np.searchsorted(edges, cc, side='right'), NB-2)
    b[np.isnan(col)] = NB-1
    bins[:,j] = b.astype(np.uint8)
print('bins done', flush=True)

def build_tree(btr, grad, resid, idx, fids, depth, min_leaf, rl):
    nodes = []
    def add(sidx, d):
        G = grad[sidx].sum(); H = float(len(sidx))
        nid = len(nodes)
        nodes.append({'leaf':True,'val':float(np.median(resid[sidx])),'f':-1,'thr':-1,'left':-1,'right':-1})
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
            if gain[jj] > 1e-6 and (best is None or gain[jj] > best[0]):
                best = (gain[jj], kk, jj)
        if best is None: return nid
        _, kk, thr = best
        mask = bs[:,kk] <= thr
        li = sidx[mask]; ri = sidx[~mask]
        if len(li)==0 or len(ri)==0: return nid
        nodes[nid].update({'leaf':False,'f':int(fids[kk]),'thr':int(thr)})
        nodes[nid]['left'] = add(li, d+1); nodes[nid]['right'] = add(ri, d+1)
        return nid
    return add(idx, 0), nodes

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

# OLS on all train rows
A = np.hstack([Z[train_mask], np.ones((train_mask.sum(),1))])
coef,*_ = np.linalg.lstsq(A, y[train_mask], rcond=None)
B = np.hstack([Z, np.ones((len(df),1))])
ols_all = B@coef
resid0 = y[train_mask] - ols_all[train_mask]
print('OLS(all-train) holdout MAE %.3f' % np.abs(ols_all[ho]-y[ho]).mean(), flush=True)

# boosting with OOB accumulation
NT = 600; SEEDS = (1,2); lr=0.05; dep=4; ml=100; rl=5.0; rs=0.75; cs=0.5
n_all = len(df); n_tr = int(train_mask.sum())
full_sum = np.zeros(n_all); oob_sum = np.zeros(n_all); oob_cnt = np.zeros(n_all)
t0 = datetime.datetime.now()
btr_full = bins[train_mask]
for s in SEEDS:
    rng = np.random.RandomState(s)
    cur = resid0.copy()
    for it in range(NT):
        grad = -np.sign(cur)
        rows = rng.rand(n_tr) < rs
        ridx = np.where(rows)[0]
        fids = np.where(rng.rand(len(feat_cols)) < cs)[0]
        if len(fids) < 10: fids = np.arange(len(feat_cols))
        root, nodes = build_tree(btr_full, grad, cur, ridx, fids, dep, ml, rl)
        pred_all = tree_predict(nodes, bins)  # all 36k rows
        full_sum += lr*pred_all
        oobm = ~rows
        oob_sum[np.where(oobm)[0]] += lr*pred_all[train_mask][oobm] if False else 0  # placeholder
        # correct OOB accumulation: indices in full array
        tr_idx = np.where(train_mask)[0]
        oob_full_pos = tr_idx[oobm]
        oob_sum[oob_full_pos] += lr*pred_all[oob_full_pos]
        oob_cnt[oob_full_pos] += 1
        cur[ridx] -= lr*pred_all[train_mask][ridx]
dt = (datetime.datetime.now()-t0).total_seconds()
print('boost done %.0fs' % dt, flush=True)
corr_full = full_sum/len(SEEDS)
corr_oob = oob_sum/np.maximum(oob_cnt,1)
print('oob_cnt min/med:', oob_cnt.min(), np.median(oob_cnt))
print('corr_full[train] std %.1f | corr_oob[train] std %.1f' % (corr_full[train_mask].std(), corr_oob[train_mask].std()))
print('corr_full[val] std %.1f' % corr_full[~train_mask].std())

# direct stacked on holdout (full-model corr)
p_direct = np.clip(ols_all + corr_full, 0, 2500)
print('direct stacked holdout MAE %.3f' % np.abs(p_direct[ho]-y[ho]).mean())

# SIMULATE evaluator: OLS on [Z, corr_oob] over train rows -> apply to holdout with corr_full
Xtr = np.hstack([Z[tr], corr_oob[tr][:,None]])
Xho = np.hstack([Z[ho], corr_full[ho][:,None]])
c,*_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
print('SIM evaluator-OLS with OOB corr: MAE %.3f' % np.abs(Xho@c - y[ho]).mean())
# what weight does corr get?
w = c[-1]; print('corr weight w=%.3f' % w)
# also ridge-ish check: with corr only (no raw features)
c2,*_ = np.linalg.lstsq(corr_oob[tr][:,None], y[tr]-0, rcond=None)
print('corr-only coef %.3f' % c2[0])

# save final table: train rows get OOB corr, val rows get full corr
out = t.copy()
out['gbm_corr'] = np.where(train_mask, corr_oob, corr_full)
print('gbm_corr quantiles:', np.quantile(out['gbm_corr'], [0,.01,.5,.99,1]).round(1))
agent_api.save_table(out, 'e014_gbm_oob.parquet')
print('saved e014_gbm_oob.parquet', out.shape)