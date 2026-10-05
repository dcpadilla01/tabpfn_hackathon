
import numpy as np, pandas as pd, collections

api = agent_api
df = api.load_saved('e011_pruned_basket.parquet')
assert df is not None, 'load failed'
tt = api.train_targets()
m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
y_all = m['future_spend_4w'].astype(float).values
print('rows', len(m), 'feats', len(feats), 'zero_share', round(float((y_all==0).mean()),4))
print(m['future_spend_4w'].describe(percentiles=[.25,.5,.75,.9,.95,.99]).to_string())

# encode features (aligned to m rows)
cols = []
for c in feats:
    s = m[c]
    if pd.api.types.is_numeric_dtype(s):
        cols.append(pd.to_numeric(s, errors='coerce'))
    else:
        cols.append(pd.Series(pd.factorize(s)[0], index=m.index).replace(-1, np.nan))
Xdf = pd.concat(cols, axis=1); Xdf.columns = feats
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)

prio = [
 ('calendar', ['snapshot_day','week']),
 ('demo', ['classification','homeowner','kid_category','has_demographics']),
 ('marketing', ['campaign','coupon']),
 ('gapzero', ['gap','zero','active_']),
 ('trend', ['decay','ewm','hl','ratio','last_year','weekly','slope','mom','trend']),
 ('mix', ['dept','brand','unit_price','distinct','n_products','variety','top']),
 ('history', ['spend','trip','days_since','tenure','lines','basket','store','max_']),
]
def grp(c):
    for g, subs in prio:
        if any(s in c for s in subs):
            return g
    return 'other'
gmap = {c: grp(c) for c in feats}
print(collections.Counter(gmap.values()))
for g in sorted(set(gmap.values())):
    print('##', g, ':', sorted([c for c in feats if gmap[c]==g]))

days_train = api.snapshot_days()['train']
H1 = [403, 431]; T1 = [d for d in days_train if d not in H1]
H2 = [347, 375, 403, 431]; T2 = [d for d in days_train if d not in H2]

def run(feat_list, train_days, hold_days, lam):
    tr = m['snapshot_day'].isin(train_days).values
    va = m['snapshot_day'].isin(hold_days).values
    Xtr = Xdf.loc[tr, feat_list].values.astype(float)
    Xva = Xdf.loc[va, feat_list].values.astype(float)
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    ytr = y_all[tr]; yva = y_all[va]
    Xb = np.hstack([Xtr, np.ones((len(Xtr),1))])
    p = Xb.shape[1]; pen = np.eye(p)*lam; pen[-1,-1]=0
    w = np.linalg.solve(Xb.T@Xb + pen, Xb.T@ytr)
    pred = np.hstack([Xva, np.ones((len(Xva),1))]) @ w
    return float(np.abs(pred-yva).mean())

lams = [10,30,100,300,1000,3000]
print('--- lambda grid (all feats, H1=[403,431]) ---')
scores = {lam: run(feats, T1, H1, lam) for lam in lams}
print({k: round(v,3) for k,v in scores.items()})
best_lam = min(scores, key=scores.get)
full = scores[best_lam]
print('best_lam', best_lam, 'full H1', round(full,3), '| full H2', round(run(feats, T2, H2, best_lam),3))

print('--- group ablation (drop group, H1) delta = MAE_drop - full ---')
for g in sorted(set(gmap.values())):
    sub = [c for c in feats if gmap[c]!=g]
    d = run(sub, T1, H1, best_lam)
    print('drop', g, round(d,3), 'delta', round(d-full,3))

# spearman correlations
sp = []
for c in feats:
    x = Xdf[c].values.astype(float)
    ok = ~np.isnan(x)
    if ok.sum()>10 and np.std(x[ok])>0:
        r = np.corrcoef(pd.Series(x[ok]).rank(), pd.Series(y_all[ok]).rank())[0,1]
    else:
        r = np.nan
    sp.append((c, r))
sp_df = pd.DataFrame(sp, columns=['feat','spearman'])
sp_df['grp'] = sp_df.feat.map(gmap)
sp_df = sp_df.sort_values('spearman')
print('--- weakest |rho| ---'); print(sp_df.head(12).to_string())
print('--- strongest ---'); print(sp_df.tail(12).to_string())
nz = sp_df[sp_df.spearman.abs()<0.015].feat.tolist()
print('n near-zero-corr feats:', len(nz))
d = run([c for c in feats if c not in nz], T1, H1, best_lam)
print('drop near-zero-corr set: MAE', round(d,3), 'delta', round(d-full,3))


# ---- cell ----

import numpy as np, pandas as pd, time
api = agent_api

def prep(path):
    df = api.load_saved(path)
    tt = api.train_targets()
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    cols = []
    for c in feats:
        s = m[c]
        if pd.api.types.is_numeric_dtype(s):
            cols.append(pd.to_numeric(s, errors='coerce').astype(float))
        else:
            cols.append(pd.Series(pd.factorize(s)[0].astype(float), index=m.index).replace(-1, np.nan))
    X = pd.concat(cols, axis=1)
    X.columns = feats
    return m, X.replace([np.inf,-np.inf], np.nan).values.astype(np.float32), feats

class GBM:
    def __init__(self, n_bins=64, max_depth=5, lr=0.08, n_trees=200, min_leaf=20, l2=1.0, seed=0):
        self.p = dict(n_bins=n_bins, max_depth=max_depth, lr=lr, n_trees=n_trees,
                      min_leaf=min_leaf, l2=l2, seed=seed)
    def _bins(self, X):
        nb = self.p['n_bins']; n, p = X.shape
        edges = []
        B = np.empty((n, p), dtype=np.int16)
        for f in range(p):
            col = X[:, f]; ok = ~np.isnan(col)
            if ok.sum() > 50:
                q = np.quantile(col[ok], np.linspace(0, 1, nb + 1)[1:-1])
                e = np.unique(q)
            else:
                e = np.array([])
            edges.append(e)
            b = np.searchsorted(e, col, side='right')
            b[~ok] = len(e)  # NaN bin = last
            B[:, f] = np.minimum(b, nb - 1).astype(np.int16) if len(e) >= nb - 1 else b.astype(np.int16)
            if len(e) < nb - 1:
                B[~ok, f] = len(e)
        self.edges_ = edges
        self.nbin_ = np.array([len(e) + 1 for e in edges])  # incl NaN bin
        return B
    def fit(self, X, y):
        P = self.p; B = self._bins(X)
        n, p = X.shape; nb_max = int(self.nbin_.max())
        rng = np.random.RandomState(P['seed'])
        pred = np.full(n, y.mean(), dtype=np.float64)
        self.trees_ = []; self.base_ = y.mean()
        min_leaf, l2, md = P['min_leaf'], P['l2'], P['max_depth']
        for t in range(P['n_trees']):
            g = pred - y
            nodes = []  # (feat, bin, left, right) or leaf
            out = np.zeros(n)
            def grow(mask, depth):
                idx = np.where(mask)[0]
                if len(idx) < 2 * min_leaf or depth >= md:
                    v = g[idx].sum() / (len(idx) + l2); nodes.append((-1, 0, -1, -1, v)); out[idx] = v; return len(nodes) - 1
                Gt = g[idx].sum(); Nt = len(idx)
                base = Gt * Gt / (Nt + l2)
                best = (0.0, -1, -1)
                for f in range(p):
                    b = B[idx, f]; nb = self.nbin_[f]
                    cw = np.bincount(b, weights=g[idx], minlength=nb).astype(np.float64)
                    cc = np.bincount(b, minlength=nb).astype(np.float64)
                    cl = np.cumsum(cw); cn = np.cumsum(cc)
                    lo = min_leaf; hi = Nt - min_leaf
                    if hi <= lo: continue
                    gl = cl[:hi]; nl = cn[:hi]
                    gr = Gt - gl; nr = Nt - nl
                    okm = nl >= lo
                    if not okm.any(): continue
                    gain = gl**2 / (nl + l2) + gr**2 / (nr + l2) - base
                    gain[~okm] = -1
                    j = int(np.argmax(gain))
                    if gain[j] > best[0]: best = (gain[j], f, j)
                gv, f, bsplit = best
                if f < 0:
                    v = Gt / (Nt + l2); nodes.append((-1, 0, -1, -1, v)); out[idx] = v; return len(nodes) - 1
                bm = B[idx, f] <= bsplit
                li = idx[bm]; ri = idx[~bm]
                if len(li) < min_leaf or len(ri) < min_leaf:
                    v = Gt / (Nt + l2); nodes.append((-1, 0, -1, -1, v)); out[idx] = v; return len(nodes) - 1
                me = len(nodes)
                nodes.append((f, bsplit, -1, -1, 0.0))
                L = grow(np.isin(np.arange(n), li) if False else _mk(li), depth + 1)
                R = grow(_mk(ri), depth + 1)
                nodes[me] = (f, bsplit, L, R, 0.0)
                return me
            # faster mask via boolean array
            msk = np.ones(n, dtype=bool)
            def grow2(mask, depth):
                idx = np.where(mask)[0]
                if len(idx) < 2 * min_leaf or depth >= md:
                    v = g[idx].sum() / (len(idx) + l2); nodes.append((-1, 0, -1, -1, v)); out[idx] = v; return len(nodes) - 1
                Gt = g[idx].sum(); Nt = len(idx); base = Gt * Gt / (Nt + l2)
                best = (0.0, -1, -1)
                for f in range(p):
                    b = B[idx, f]; nb = self.nbin_[f]
                    cw = np.bincount(b, weights=g[idx], minlength=nb)
                    cc = np.bincount(b, minlength=nb)
                    cl = np.cumsum(cw); cn = np.cumsum(cc)
                    hi = Nt - min_leaf
                    if hi <= min_leaf: continue
                    gl = cl[:hi]; nl = cn[:hi]; gr = Gt - gl; nr = Nt - nl
                    gain = gl**2 / (nl + l2) + gr**2 / (nr + l2) - base
                    gain[nl < min_leaf] = -1
                    j = int(np.argmax(gain))
                    if gain[j] > best[0]: best = (float(gain[j]), f, j)
                gv, f, bsplit = best
                if f < 0:
                    v = Gt / (Nt + l2); nodes.append((-1, 0, -1, -1, v)); out[idx] = v; return len(nodes) - 1
                bm = B[idx, f] <= bsplit
                li = idx[bm]; ri = idx[~bm]
                if len(li) < min_leaf or len(ri) < min_leaf:
                    v = Gt / (Nt + l2); nodes.append((-1, 0, -1, -1, v)); out[idx] = v; return len(nodes) - 1
                me = len(nodes); nodes.append((f, bsplit, -1, -1, 0.0))
                m2 = mask.copy(); m2[li] = False
                L = grow2(mask & bm if False else np.isin(np.arange(n), li), depth + 1)
                return me
            # simpler: use boolean masks directly
            nodes.clear(); out[:] = 0
            def grow3(mask, depth):
                idx = np.where(mask)[0]
                if len(idx) < 2 * min_leaf or depth >= md:
                    v = g[idx].sum() / (len(idx) + l2); nodes.append((-1, 0, -1, -1, v)); out[idx] = v; return len(nodes) - 1
                Gt = g[idx].sum(); Nt = len(idx); base = Gt * Gt / (Nt + l2)
                best = (0.0, -1, -1)
                for f in range(p):
                    b = B[idx, f]; nb = self.nbin_[f]
                    cw = np.bincount(b, weights=g[idx], minlength=nb)
                    cc = np.bincount(b, minlength=nb)
                    cl = np.cumsum(cw); cn = np.cumsum(cc)
                    hi = Nt - min_leaf
                    if hi <= min_leaf: continue
                    gl = cl[:hi]; nl = cn[:hi]; gr = Gt - gl; nr = Nt - nl
                    gain = gl**2 / (nl + l2) + gr**2 / (nr + l2) - base
                    gain[nl < min_leaf] = -1
                    j = int(np.argmax(gain))
                    if gain[j] > best[0]: best = (float(gain[j]), f, j)
                gv, f, bsplit = best
                if f < 0:
                    v = Gt / (Nt + l2); nodes.append((-1, 0, -1, -1, v)); out[idx] = v; return len(nodes) - 1
                bm = np.zeros(n, dtype=bool); bm[idx] = B[idx, f] <= bsplit
                li = idx[bm[idx]]; ri = idx[~bm[idx]]
                if len(li) < min_leaf or len(ri) < min_leaf:
                    v = Gt / (Nt + l2); nodes.append((-1, 0, -1, -1, v)); out[idx] = v; return len(nodes) - 1
                me = len(nodes); nodes.append((f, bsplit, -1, -1, 0.0))
                L = grow3(bm, depth + 1)
                R = grow3(mask & ~bm, depth + 1)
                nodes[me] = (f, bsplit, L, R, 0.0)
                return me
            grow3(msk, 0)
            self.trees_.append(nodes)
            pred += P['lr'] * out
        return self
    def predict(self, X):
        B = np.empty(X.shape, dtype=np.int16)
        for f, e in enumerate(self.edges_):
            col = X[:, f]; ok = ~np.isnan(col)
            b = np.searchsorted(e, col, side='right'); b[~ok] = len(e)
            B[:, f] = b
        pred = np.full(len(X), self.base_)
        for nodes in self.trees_:
            out = np.zeros(len(X))
            def apply(ni, mask):
                f, bsplit, L, R, v = nodes[ni]
                if f < 0: out[mask] = v; return
                go_l = np.zeros(len(X), dtype=bool); go_l[mask] = B[mask, f] <= bsplit
                apply(L, mask & go_l); apply(R, mask & ~go_l)
            apply(0, np.ones(len(X), dtype=bool))
            pred += self.p['lr'] * out
        return pred

def run_gbm(X, y, days, train_days, hold_days, **kw):
    tr = np.isin(days, train_days); va = np.isin(days, hold_days)
    mdl = GBM(**kw).fit(X[tr], y[tr].astype(np.float64))
    pv = mdl.predict(X[va])
    return float(np.abs(pv - y[va]).mean()), mdl

t0 = time.time()
m, X, feats = prep('e011_pruned_basket.parquet')
y = m['future_spend_4w'].values.astype(np.float64)
days = m['snapshot_day'].values
days_train = api.snapshot_days()['train']
H1 = [403, 431]; T1 = [d for d in days_train if d not in H1]
H2 = [347, 375, 403, 431]; T2 = [d for d in days_train if d not in H2]
print('prep done', round(time.time()-t0,1), 's; X', X.shape)
for kw in [dict(n_trees=150, lr=0.08, max_depth=5), dict(n_trees=300, lr=0.05, max_depth=6)]:
    t = time.time()
    s1, _ = run_gbm(X, y, days, T1, H1, **kw)
    s2, _ = run_gbm(X, y, days, T2, H2, **kw)
    print(kw, 'H1', round(s1,3), 'H2', round(s2,3), 'time', round(time.time()-t,1))
# baselines
tr1 = np.isin(days, T1); va1 = np.isin(days, H1)
print('mean-base H1', round(float(np.abs(y[tr1].mean()-y[va1]).mean()),3))
i28 = feats.index('spend_28')
print('spend_28-base H1', round(float(np.abs(np.nan_to_num(X[va1,i28]) - y[va1]).mean()),3))
