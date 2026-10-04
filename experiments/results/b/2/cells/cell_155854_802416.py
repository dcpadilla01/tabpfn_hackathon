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
