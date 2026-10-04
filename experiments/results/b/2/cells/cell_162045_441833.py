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
