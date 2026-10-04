import agent_api, pandas as pd, numpy as np
pd.set_option('display.width', 250)

t = agent_api.load_saved('e011_price.parquet')
print('E011 shape:', t.shape)
print(t.dtypes.value_counts())
nonnum = [c for c in t.columns if not pd.api.types.is_numeric_dtype(t[c])]
print('non-numeric:', nonnum)
cols = list(t.columns)
for i in range(0, len(cols), 8):
    print(' | '.join(cols[i:i+8]))

tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged:', df.shape)
yv = df['future_spend_4w'].values.astype(float)
print('target describe:'); print(df['future_spend_4w'].describe())
print('zero share:', float((df.future_spend_4w==0).mean()))
print('rows per snapshot:'); print(df.groupby('snapshot_day').size().to_dict())

feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
print('overall NaN frac:', round(float(X.isna().mean().mean()),4))

tr403 = (df.snapshot_day<=403).values; va431 = (df.snapshot_day==431).values
tr375 = (df.snapshot_day<=375).values; va403 = (df.snapshot_day==403).values

mu = X[tr403].mean(); sd = X[tr403].std().replace(0,1.0)
Z = ((X-mu)/sd).fillna(0.0).values

def mae_for(a, trm, vam):
    A = Z[trm].T@Z[trm] + a*np.eye(Z.shape[1])
    b = Z[trm].T@yv[trm]
    w = np.linalg.solve(A,b)
    p = Z[vam]@w
    return float(np.abs(p-yv[vam]).mean())

print('--- ridge alpha sweep, val=431 (train<=403) ---')
for a in [1,3,10,30,100,300,1000,3000]:
    print(a, round(mae_for(a, tr403, va431),3))
print('--- val=403 (train<=375) ---')
for a in [10,30,100,300,1000]:
    print(a, round(mae_for(a, tr375, va403),3))

print('naive mean pred MAE431:', round(float(np.abs(yv[tr403].mean()-yv[va431]).mean()),3))

Xtr = X[tr403].copy(); ytr = pd.Series(yv[tr403])
corr = Xtr.corrwith(ytr)
top = corr.abs().sort_values(ascending=False).head(30)
print('top |corr| features:')
for c in top.index: print(f'  {c}: {corr[c]:+.3f}')

c28 = [c for c in X.columns if 'spend' in c.lower() and '28' in c]
print('spend28 candidates:', c28)
if c28:
    x = X[c28[0]].values
    m = tr403 & ~np.isnan(x)
    b_, a_ = np.polyfit(x[m], yv[m], 1)
    mv = va431 & ~np.isnan(x)
    p = a_ + b_*x[mv]
    print('lin spend28 MAE431:', round(float(np.abs(p-yv[mv]).mean()),3), 'slope', round(b_,3), 'icept', round(a_,2))

v = agent_api.snapshot()
txv = v.transactions
print('tx shape', txv.shape)
print(txv.head(3))
print('neg sales lines:', int((txv.sales_value<0).sum()), 'neg qty:', int((txv.quantity<0).sum()))
print('households attr:', type(v.households), (v.households.shape if hasattr(v.households,'shape') else len(v.households)))
print('day', v.day, 'week', v.week)
print('snapshot_days:', agent_api.snapshot_days())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e011_price.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
yv = df['future_spend_4w'].values.astype(float)
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
tr403 = (df.snapshot_day<=403).values; va431=(df.snapshot_day==431).values
tr375 = (df.snapshot_day<=375).values; va403=(df.snapshot_day==403).values
mu = X[tr403].mean(); sd = X[tr403].std().replace(0,1.0)
Z = ((X-mu)/sd).fillna(0.0).values
def mae_for(a, trm, vam):
    A = Z[trm].T@Z[trm] + a*np.eye(Z.shape[1]); b = Z[trm].T@yv[trm]
    w = np.linalg.solve(A,b); p = Z[vam]@w
    return float(np.abs(p-yv[vam]).mean())
print('val431:', [(a, round(mae_for(a,tr403,va431),3)) for a in [3,10,30,100,300,1000,3000]])
print('val403:', [(a, round(mae_for(a,tr375,va403),3)) for a in [10,30,100,300,1000]])
print('mean-pred MAE431:', round(float(np.abs(yv[tr403].mean()-yv[va431]).mean()),3))
print('median-pred MAE431:', round(float(np.abs(np.median(yv[tr403])-yv[va431]).mean()),3))
print('target describe:', {k: round(float(v),2) for k,v in df.future_spend_4w.describe().items()})
print('zero share:', round(float((df.future_spend_4w==0).mean()),4))
print('rows/snap:', df.groupby('snapshot_day').size().to_dict())
v = agent_api.snapshot()
print('day', v.day, 'week', v.week)
print('households type:', type(v.households))
tx = v.transactions
print('tx', tx.shape)
print('neg sales lines:', int((tx.sales_value<0).sum()), 'neg qty:', int((tx.quantity<0).sum()))
print('sales describe:', {k: round(float(vv),2) for k,vv in tx.sales_value.describe().items()})


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e011_price.parquet')
tt = agent_api.train_targets()
print('tt shape', tt.shape, 'dups:', int(tt.duplicated(['household_key','snapshot_day']).sum()))
print('tt snapshots:', sorted(tt.snapshot_day.unique()))
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape, 'dups:', int(df.duplicated(['household_key','snapshot_day']).sum()))
g = df.groupby('snapshot_day')['future_spend_4w']
stats = pd.DataFrame({'n': g.size(), 'mean': g.mean(), 'med': g.median(), 'p90': g.quantile(.9), 'max': g.max(), 'zero': g.apply(lambda s:(s==0).mean())})
print(stats.round(1))
# simple baselines per snapshot: predict global train mean / median
yv = df.future_spend_4w.values
for sd in [403, 431]:
    m = df.snapshot_day==sd
    tr = df.snapshot_day<sd
    print(f'day {sd}: mean-pred MAE {np.abs(yv[tr].mean()-yv[m]).mean():.2f}, med-pred {np.abs(np.median(yv[tr])-yv[m]).mean():.2f}, top5 targets {np.sort(yv[m])[-5:]}')
# check huge targets
big = df.nlargest(10,'future_spend_4w')[['household_key','snapshot_day','future_spend_4w','spend28','spend364','lt_spend']]
print(big)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e011_price.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
yv = df.future_spend_4w.values.astype(float)
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
x = df.spend28.values.astype(float)
m = tr & ~np.isnan(x)
b_, a_ = np.polyfit(x[m], yv[m], 1)
mv = va & ~np.isnan(x)
p = a_ + b_*x[mv]
print('univar spend28: corr', round(float(np.corrcoef(x[m], yv[m])[0,1]),3), 'slope', round(b_,3), 'MAE431', round(float(np.abs(p-yv[mv]).mean()),3))
# clipped slope (spend can't be huge)
b2, a2 = np.polyfit(np.minimum(x[m], 400), yv[m], 1)
p2 = a2 + b2*np.minimum(x[mv],400)
print('clipped400 univar MAE431', round(float(np.abs(p2-yv[mv]).mean()),3))
# binned means: 25 quantile bins of spend28 on train, predict bin mean
qs = np.quantile(x[m], np.linspace(0,1,26))
bins = np.digitize(x[mv], qs[1:-1])
bmean = np.array([yv[m][(np.digitize(x[m],qs[1:-1])==k)].mean() for k in range(25)])
pb = bmean[bins]
print('binned25 spend28 MAE431', round(float(np.abs(pb-yv[mv]).mean()),3))
# two-feature: spend28 + lag trend
x2 = df.lag_mean_1_4.values.astype(float)
mm = tr & ~np.isnan(x) & ~np.isnan(x2)
A = np.c_[np.ones(mm.sum()), x[mm], x2[mm]]
w = np.linalg.lstsq(A, yv[mm], rcond=None)[0]
mv2 = va & ~np.isnan(x) & ~np.isnan(x2)
p3 = w[0]+w[1]*x[mv2]+w[2]*x2[mv2]
print('spend28+lagmean14 MAE431', round(float(np.abs(p3-yv[mv2]).mean()),3))
# per-snapshot corr of spend28 with y
for sd in [95,207,291,375,403,431]:
    mm2 = df.snapshot_day==sd
    print('corr@',sd, round(float(np.corrcoef(x[mm2.values], yv[mm2.values])[0,1]),3))
# what does the best possible constant-mixture do: predict y = spend28 exactly? MAE of |y - spend28|
print('MAE if pred=spend28:', round(float(np.abs(x[mv]-yv[mv]).mean()),3))
print('MAE if pred=0.8*spend28:', round(float(np.abs(0.8*x[mv]-yv[mv]).mean()),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

NB = 32; NANB = NB-1

def bin_matrix(Xtr_cols_dict, df, feats, trm):
    B = np.zeros((len(df), len(feats)), dtype=np.uint8)
    edges = []
    for j,c in enumerate(feats):
        x = df[c].values.astype(float)
        e = np.unique(np.quantile(x[trm], np.linspace(0,1,NB)))
        if len(e) <= 2:
            b = (x > e[0]).astype(np.int8)
        else:
            b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
        B[:,j] = np.where(np.isnan(x), NANB, np.clip(b, 0, NANB))
        edges.append(e)
    return B, edges

def build_tree(B, rows, r, ci, depth, min_leaf, lam, rng):
    # returns list of nodes: (feat_j, thr_bin, left_rows, right_rows) via recursion
    tree = {}
    def rec(R, d):
        if d >= depth or len(R) < 2*min_leaf:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        sub = B[np.ix_(R, ci)]
        idx = sub.astype(np.int32) + NB*np.arange(len(ci))[None,:]
        w = np.repeat(r[R][:,None], len(ci), axis=1).ravel()
        H = np.bincount(idx.ravel(), minlength=NB*len(ci)).reshape(len(ci), NB).astype(float)
        G = np.bincount(idx.ravel(), weights=w, minlength=NB*len(ci)).reshape(len(ci), NB)
        cg = np.cumsum(G, axis=1)[:,:-1]; ch = np.cumsum(H, axis=1)[:,:-1]
        Gt = G.sum(1)[:,None]; Ht = H.sum(1)[:,None]
        gl, gr = cg, Gt-cg; hl, hr = ch, Ht-ch
        gain = gl**2/(hl+lam) + gr**2/(hr+lam) - Gt**2/(Ht+lam)
        gain[hl<min_leaf] = -1; gain[hr<min_leaf] = -1
        fj, bj = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[fj,bj] <= 1e-9:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        f = ci[fj]
        bcol = B[R, f]
        L = R[bcol <= bj]; Rr = R[bcol > bj]
        if len(L)==0 or len(Rr)==0:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        nid = len(tree)
        tree[nid] = (f, bj, None, None)
        tree[nid] = (f, bj, rec(L, d+1), rec(Rr, d+1))
        return ('N', nid)
    root = rec(rows, 0)
    return root, tree

def apply_tree(root, tree, B, rows):
    out = np.zeros(len(rows))
    stack = [(root, np.arange(len(rows)))]
    while stack:
        node, idx = stack.pop()
        if node[0]=='L':
            out[idx] = node[1]; continue
        nid = node[1]; f, bj, L, Rr = tree[nid]
        bcol = B[rows[idx], f]
        li = idx[bcol <= bj]; ri = idx[bcol > bj]
        stack.append((L, li)); stack.append((Rr, ri))
    return out

def gbm(df, feats, trm, vam, n_trees=150, lr=0.12, depth=4, min_leaf=50, lam=3.0, clip_q=0.97, seed=1, ncol=40, rsub=0.6, verbose=False):
    y = df.future_spend_4w.values.astype(float)
    yw = np.clip(y, 0, np.quantile(y[trm], clip_q))
    B, edges = bin_matrix(None, df, feats, trm)
    rng = np.random.RandomState(seed)
    tr_idx = np.where(trm)[0]; va_idx = np.where(vam)[0]
    F = np.zeros(len(y)); Fv = np.zeros(len(y))
    base = yw[trm].mean(); F[:] = 0; Fv[:] = 0
    for t in range(n_trees):
        samp = tr_idx[rng.rand(len(tr_idx)) < rsub]
        ci = np.where(rng.rand(len(feats)) < ncol/len(feats))[0]
        if len(ci) < 5: ci = np.arange(len(feats))
        r = yw - F - base
        root, tree = build_tree(B, samp, r, ci, depth, min_leaf, lam, rng)
        Fv += base*0 + 0  # base added once outside
        Fv[va_idx] += apply_tree(root, tree, B, va_idx)
        F[tr_idx] += apply_tree(root, tree, B, tr_idx)
    p = base + Fv[va_idx]
    p = np.clip(p, 0, None)
    return p, y[vam]

df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
core = ['spend28','trips28','recency','spend56','spend112','spend364','tenure','lt_spend']
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
p, y = gbm(df, core, tr, va)
print('proxy GBM core8 MAE431:', round(float(np.abs(p-y).mean()),3))
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
p, y = gbm(df, feats, tr, va)
print('proxy GBM all228 MAE431:', round(float(np.abs(p-y).mean()),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

NB = 32; NANB = NB-1

def bin_matrix(df, feats, trm):
    B = np.zeros((len(df), len(feats)), dtype=np.uint8)
    for j,c in enumerate(feats):
        x = df[c].values.astype(float)
        xv = x[trm]
        if np.all(np.isnan(xv)):
            B[:,j] = NANB; continue
        e = np.unique(np.quantile(xv, np.linspace(0,1,NB)))
        if len(e) <= 2:
            b = (x > e[0]).astype(np.int8)
        else:
            b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
        B[:,j] = np.where(np.isnan(x), NANB, np.clip(b, 0, NANB))
    return B

def build_tree(B, rows, r, ci, depth, min_leaf, lam):
    tree = {}
    def rec(R, d):
        if d >= depth or len(R) < 2*min_leaf:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        sub = B[np.ix_(R, ci)]
        idx = sub.astype(np.int32) + NB*np.arange(len(ci))[None,:]
        w = np.repeat(r[R][:,None], len(ci), axis=1).ravel()
        H = np.bincount(idx.ravel(), minlength=NB*len(ci)).reshape(len(ci), NB).astype(float)
        G = np.bincount(idx.ravel(), weights=w, minlength=NB*len(ci)).reshape(len(ci), NB)
        cg = np.cumsum(G, axis=1)[:,:-1]; ch = np.cumsum(H, axis=1)[:,:-1]
        Gt = G.sum(1)[:,None]; Ht = H.sum(1)[:,None]
        gl, gr = cg, Gt-cg; hl, hr = ch, Ht-ch
        gain = gl**2/(hl+lam) + gr**2/(hr+lam) - Gt**2/(Ht+lam)
        gain[hl<min_leaf] = -1; gain[hr<min_leaf] = -1
        fj, bj = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[fj,bj] <= 1e-9:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        f = ci[fj]
        bcol = B[R, f]
        L = R[bcol <= bj]; Rr = R[bcol > bj]
        if len(L)==0 or len(Rr)==0:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        nid = len(tree)
        left = rec(L, d+1); right = rec(Rr, d+1)
        tree[nid] = (f, bj, left, right)
        return ('N', nid)
    return rec(rows, 0), tree

def apply_tree(root, tree, B, rows):
    out = np.zeros(len(rows))
    stack = [(root, np.arange(len(rows)))]
    while stack:
        node, idx = stack.pop()
        if node[0]=='L':
            out[idx] = node[1]; continue
        f, bj, L, Rr = tree[node[1]]
        bcol = B[rows[idx], f]
        stack.append((L, idx[bcol <= bj])); stack.append((Rr, idx[bcol > bj]))
    return out

def gbm(df, feats, trm, vam, n_trees=120, lr=0.12, depth=4, min_leaf=60, lam=5.0, clip_q=0.97, seed=1, ncol=40, rsub=0.6, dbg=False):
    y = df.future_spend_4w.values.astype(float)
    yw = np.clip(y, 0, np.quantile(y[trm], clip_q))
    B = bin_matrix(df, feats, trm)
    if dbg:
        print('B stats: min', B.min(), 'max', B.max(), 'nan?', np.isnan(B).any())
    rng = np.random.RandomState(seed)
    tr_idx = np.where(trm)[0]; va_idx = np.where(vam)[0]
    base = yw[trm].mean()
    F = np.zeros(len(y)); Fv = np.zeros(len(y))
    for t in range(n_trees):
        samp = tr_idx[rng.rand(len(tr_idx)) < rsub]
        ci = np.where(rng.rand(len(feats)) < ncol/len(feats))[0]
        if len(ci) < 5: ci = np.arange(len(feats))
        r = yw - F - base
        if dbg and t==0: print('resid0 stats:', r.min(), r.max(), r.mean())
        root, tree = build_tree(B, samp, r, ci, depth, min_leaf, lam)
        out_tr = apply_tree(root, tree, B, tr_idx)
        if dbg and t==0: print('tree0 out stats:', out_tr.min(), out_tr.max())
        Fv[va_idx] += apply_tree(root, tree, B, va_idx)
        F[tr_idx] += out_tr
    p = np.clip(base + Fv[va_idx], 0, None)
    return p, y[vam]

df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
core = ['spend28','trips28','recency','spend56','spend112','spend364','tenure','lt_spend']
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
p, y = gbm(df, core, tr, va, dbg=True)
print('proxy GBM core8 MAE431:', round(float(np.abs(p-y).mean()),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
from datetime import datetime as dt
T0 = dt.now()
def ts(msg):
    print(msg, (dt.now()-T0).total_seconds())

df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
tr = (df.snapshot_day<=403).values
ts('loaded')

NB = 32; NANB = NB-1
def bin_matrix(df, feats, trm):
    B = np.zeros((len(df), len(feats)), dtype=np.uint8)
    for j,c in enumerate(feats):
        x = df[c].values.astype(float)
        xv = x[trm]
        if np.all(np.isnan(xv)):
            B[:,j] = NANB; continue
        e = np.unique(np.quantile(xv, np.linspace(0,1,NB)))
        if len(e) <= 2:
            b = (x > e[0]).astype(np.int8)
        else:
            b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
        B[:,j] = np.where(np.isnan(x), NANB, np.clip(b, 0, NANB))
    return B

feats = ['spend28','trips28','recency','spend56','spend112','spend364','tenure','lt_spend']
ts('start binning')
B = bin_matrix(df, feats, tr)
ts('binned')

def build_tree(B, rows, r, ci, depth, min_leaf, lam):
    tree = {}
    def rec(R, d):
        if d >= depth or len(R) < 2*min_leaf:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        sub = B[np.ix_(R, ci)]
        idx = sub.astype(np.int32) + NB*np.arange(len(ci))[None,:]
        w = np.repeat(r[R][:,None], len(ci), axis=1).ravel()
        H = np.bincount(idx.ravel(), minlength=NB*len(ci)).reshape(len(ci), NB).astype(float)
        G = np.bincount(idx.ravel(), weights=w, minlength=NB*len(ci)).reshape(len(ci), NB)
        cg = np.cumsum(G, axis=1)[:,:-1]; ch = np.cumsum(H, axis=1)[:,:-1]
        Gt = G.sum(1)[:,None]; Ht = H.sum(1)[:,None]
        gl, gr = cg, Gt-cg; hl, hr = ch, Ht-ch
        gain = gl**2/(hl+lam) + gr**2/(hr+lam) - Gt**2/(Ht+lam)
        gain[hl<min_leaf] = -1; gain[hr<min_leaf] = -1
        fj, bj = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[fj,bj] <= 1e-9:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        f = ci[fj]
        bcol = B[R, f]
        L = R[bcol <= bj]; Rr = R[bcol > bj]
        if len(L)==0 or len(Rr)==0:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        nid = len(tree)
        left = rec(L, d+1); right = rec(Rr, d+1)
        tree[nid] = (f, bj, left, right)
        return ('N', nid)
    return rec(rows, 0), tree

def apply_tree(root, tree, B, rows):
    out = np.zeros(len(rows))
    stack = [(root, np.arange(len(rows)))]
    while stack:
        node, idx = stack.pop()
        if node[0]=='L':
            out[idx] = node[1]; continue
        f, bj, L, Rr = tree[node[1]]
        bcol = B[rows[idx], f]
        stack.append((L, idx[bcol <= bj])); stack.append((Rr, idx[bcol > bj]))
    return out

y = df.future_spend_4w.values.astype(float)
yw = np.clip(y, 0, np.quantile(y[tr], 0.97))
rng = np.random.RandomState(1)
tr_idx = np.where(tr)[0]
base = yw[tr].mean()
F = np.zeros(len(y))
for t in range(5):
    samp = tr_idx[rng.rand(len(tr_idx)) < 0.6]
    ci = np.arange(len(feats))
    r = yw - F - base
    ts(f'tree {t} start')
    root, tree = build_tree(B, samp, r, ci, 4, 60, 5.0)
    ts(f'tree {t} built, nodes={len(tree)}')
    F += apply_tree(root, tree, B, tr_idx)
    ts(f'tree {t} applied')
print('F stats', F.min(), F.max())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
print('start', flush=True)
df = agent_api.load_saved('e011_price.parquet')
print('loaded', df.shape, flush=True)
tt = agent_api.train_targets()
print('targets', tt.shape, flush=True)
m = df.merge(tt, on=['household_key','snapshot_day'])
print('merged', m.shape, flush=True)
x = m.spend28.values.astype(float)
print('col ok', np.nanmean(x), flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
print('start', flush=True)
df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
tr = (df.snapshot_day<=403).values
NB = 32
feats = ['spend28','trips28','recency']
B = np.zeros((len(df), len(feats)), dtype=np.uint8)
for j,c in enumerate(feats):
    x = df[c].values.astype(float)
    e = np.unique(np.quantile(x[tr], np.linspace(0,1,NB)))
    b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
    B[:,j] = np.where(np.isnan(x), NB-1, np.clip(b,0,NB-1))
print('binned', flush=True)
rows = np.where(tr)[0]
r = df.future_spend_4w.values.astype(float); r = np.clip(r,0,300); r = r - r[tr].mean()
ci = np.arange(len(feats))
sub = B[np.ix_(rows, ci)]
print('sub', sub.shape, sub.dtype, flush=True)
idx = sub.astype(np.int32) + NB*np.arange(len(ci))[None,:]
print('idx done', idx.shape, idx.max(), flush=True)
w = np.repeat(r[rows][:,None], len(ci), axis=1).ravel()
print('w done', flush=True)
H = np.bincount(idx.ravel(), minlength=NB*len(ci)).reshape(len(ci), NB).astype(float)
print('H done', flush=True)
G = np.bincount(idx.ravel(), weights=w, minlength=NB*len(ci)).reshape(len(ci), NB)
print('G done', H.sum(), G.sum(), flush=True)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
from datetime import datetime as dt
T0 = dt.now()
def ts(msg): print(msg, round((dt.now()-T0).total_seconds(),1), 's', flush=True)

df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
NB = 32; NANB = NB-1

def bin_matrix(df, feats, trm):
    B = np.zeros((len(df), len(feats)), dtype=np.uint8)
    for j,c in enumerate(feats):
        x = df[c].values.astype(float)
        xv = x[trm]
        if np.all(np.isnan(xv)):
            B[:,j] = NANB; continue
        e = np.unique(np.quantile(xv, np.linspace(0,1,NB)))
        if len(e) <= 2:
            b = (x > e[0]).astype(np.int8)
        else:
            b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
        B[:,j] = np.where(np.isnan(x), NANB, np.clip(b,0,NANB))
    return B

feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
ts('start binning %d feats' % len(feats))
B = bin_matrix(df, feats, tr)
ts('binned')

def build_tree(B, rows, r, ci, depth, min_leaf, lam):
    tree = {}
    def rec(R, d):
        if d >= depth or len(R) < 2*min_leaf or len(R)==0:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam) if len(R)>0 else 0.0)
        sub = B[np.ix_(R, ci)]
        idx = sub.astype(np.int32) + NB*np.arange(len(ci))[None,:]
        w = np.repeat(r[R][:,None], len(ci), axis=1).ravel()
        H = np.bincount(idx.ravel(), minlength=NB*len(ci)).reshape(len(ci), NB).astype(float)
        G = np.bincount(idx.ravel(), weights=w, minlength=NB*len(ci)).reshape(len(ci), NB)
        cg = np.cumsum(G, axis=1)[:,:-1]; ch = np.cumsum(H, axis=1)[:,:-1]
        Gt = G.sum(1)[:,None]; Ht = H.sum(1)[:,None]
        gl, gr = cg, Gt-cg; hl, hr = ch, Ht-ch
        gain = gl**2/(hl+lam) + gr**2/(hr+lam) - Gt**2/(Ht+lam)
        gain[hl<min_leaf] = -1; gain[hr<min_leaf] = -1
        fj, bj = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[fj,bj] <= 1e-9:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        f = ci[fj]
        bcol = B[R, f]
        L = R[bcol <= bj]; Rr = R[bcol > bj]
        if len(L)==0 or len(Rr)==0:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        nid = len(tree)
        left = rec(L, d+1); right = rec(Rr, d+1)
        tree[nid] = (f, bj, left, right)
        return ('N', nid)
    return rec(rows, 0), tree

def apply_tree(root, tree, B, rows):
    out = np.zeros(len(rows))
    stack = [(root, np.arange(len(rows)))]
    while stack:
        node, idx = stack.pop()
        if node[0]=='L':
            out[idx] = node[1]; continue
        f, bj, L, Rr = tree[node[1]]
        bcol = B[rows[idx], f]
        stack.append((L, idx[bcol <= bj])); stack.append((Rr, idx[bcol > bj]))
    return out

y = df.future_spend_4w.values.astype(float)
yw = np.clip(y, 0, np.quantile(y[tr], 0.97))
rng = np.random.RandomState(1)
tr_idx = np.where(tr)[0]; va_idx = np.where(va)[0]
base = yw[tr].mean()
F = np.zeros(len(y)); Fv = np.zeros(len(y))
for t in range(10):
    samp = tr_idx[rng.rand(len(tr_idx)) < 0.6]
    ci = np.where(rng.rand(len(feats)) < 40/len(feats))[0]
    if len(ci) < 5: ci = np.arange(len(feats))
    r = yw - F - base
    root, tree = build_tree(B, samp, r, ci, 4, 60, 5.0)
    F[tr_idx] += apply_tree(root, tree, B, tr_idx)
    Fv[va_idx] += apply_tree(root, tree, B, va_idx)
    if t%5==4: ts('tree %d done' % t)
p = np.clip(base + Fv[va_idx], 0, None)
print('MAE431 (10 trees):', round(float(np.abs(p - y[va_idx]).mean()),3), flush=True)
ts('done')


# ---- cell ----
import agent_api, pandas as pd, numpy as np
df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
NB = 32; NANB = NB-1
feats = ['spend28','trips28','recency','spend56','spend112','spend364','tenure','lt_spend']
B = np.zeros((len(df), len(feats)), dtype=np.uint8)
for j,c in enumerate(feats):
    x = df[c].values.astype(float)
    e = np.unique(np.quantile(x[tr], np.linspace(0,1,NB)))
    b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
    B[:,j] = np.where(np.isnan(x), NANB, np.clip(b,0,NANB))
y = df.future_spend_4w.values.astype(float)
yw = np.clip(y, 0, np.quantile(y[tr], 0.97))
r = yw - yw[tr].mean()
rows = np.where(tr)[0]
ci = np.arange(len(feats))
sub = B[np.ix_(rows, ci)]
idx = sub.astype(np.int32) + NB*np.arange(len(ci))[None,:]
w = np.repeat(r[rows][:,None], len(ci), axis=1).ravel()
H = np.bincount(idx.ravel(), minlength=NB*len(ci)).reshape(len(ci), NB).astype(float)
G = np.bincount(idx.ravel(), weights=w, minlength=NB*len(ci)).reshape(len(ci), NB)
cg = np.cumsum(G, axis=1)[:,:-1]; ch = np.cumsum(H, axis=1)[:,:-1]
Gt = G.sum(1)[:,None]; Ht = H.sum(1)[:,None]
gain = cg**2/(ch+5.0) + (Gt-cg)**2/(Ht-ch+5.0) - Gt**2/(Ht+5.0)
gain[ch<60] = -1; gain[Ht-ch<60] = -1
fj, bj = np.unravel_index(np.argmax(gain), gain.shape)
print('root split: feat', feats[fj], 'bin', bj, 'gain', gain[fj,bj])
f = fj
bcol = B[rows, f]
L = rows[bcol <= bj]; Rr = rows[bcol > bj]
print('split sizes', len(L), len(Rr))
print('leaf vals', r[L].mean(), r[Rr].mean())
va_idx = np.where(va)[0]
pred = np.where(B[va_idx, f] <= bj, r[L].mean(), r[Rr].mean()) + yw[tr].mean()
print('1-split stump MAE431:', round(float(np.abs(pred - y[va_idx]).mean()),3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def make_feats(view, snapshot_day):
    tx = view.transactions
    hh = sorted(set(view.households))
    idx = pd.Index(hh, name='household_key')
    g = tx.groupby('household_key')
    first = g['day'].min().reindex(hh)
    lt_spend = g['sales_value'].sum().reindex(hh)
    lt_trips = g['basket_id'].nunique().reindex(hh)
    ten = (snapshot_day - first + 1).clip(lower=1)
    out = pd.DataFrame(index=idx)
    out['lt_spend_per_day'] = lt_spend/ten
    out['lt_trips_per_day'] = lt_trips/ten
    out['lt_spend_per_trip'] = lt_spend/lt_trips.clip(lower=1)
    tx364 = tx[tx.day > snapshot_day-364]
    g3 = tx364.groupby('household_key')
    sp3 = g3['sales_value'].sum().reindex(hh).fillna(0.0)
    tr3 = g3['basket_id'].nunique().reindex(hh).fillna(0.0)
    out['spend_per_day364'] = sp3/364.0
    out['spend_per_trip364'] = sp3/tr3.clip(lower=1)
    # weekly CV over last 52 weeks (zero weeks included)
    w0 = snapshot_day//7
    weeks = np.arange(max(w0-51,0), w0+1)
    tw = tx.assign(wk=(tx.day//7).astype(int))
    wsum = tw[tw.wk.isin(weeks)].groupby(['household_key','wk'])['sales_value'].sum().unstack(fill_value=0.0)
    wsum = wsum.reindex(index=hh, columns=weeks).fillna(0.0)
    mu = wsum.mean(axis=1); sd = wsum.std(axis=1)
    out['cv_weekly_52'] = np.where(mu>0, sd/np.maximum(mu,1e-9), np.nan)
    # 4-week windows over past year
    L = {}
    for k in range(1,14):
        hi = snapshot_day-28*(k-1); lo = snapshot_day-28*k
        msk = (tx.day>lo)&(tx.day<=hi)
        L[k] = tx[msk].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    Ldf = pd.DataFrame(L)
    m1 = Ldf.mean(axis=1); s1 = Ldf.std(axis=1)
    out['cv_window_1y'] = np.where(m1>0, s1/np.maximum(m1,1e-9), np.nan)
    # gaps / overdue
    last = g['day'].max().reindex(hh)
    rec = (snapshot_day-last).astype(float)
    tx112 = tx[tx.day > snapshot_day-112]
    bd = tx112.groupby('household_key')['day'].apply(lambda s: np.sort(pd.unique(s.values)))
    def gs(a):
        if a is None or len(a)<2: return (np.nan, np.nan)
        d = np.diff(a); return (float(d.max()), float(np.median(d)))
    g2 = bd.apply(gs)
    gmed = g2.apply(lambda t: t[1])
    out['max_gap_112'] = g2.apply(lambda t: t[0])
    out['overdue_days'] = rec - gmed
    out['overdue_ratio'] = rec/(gmed.fillna(3)+1.0)
    # top store health
    ts112 = tx112.groupby(['household_key','store_id']).size().rename('n').reset_index()
    topmap = ts112.loc[ts112.groupby('household_key')['n'].idxmax()].set_index('household_key')
    tot = ts112.groupby('household_key')['n'].sum()
    out['top_store_share112'] = (topmap['n']/tot)
    store_last = tx.groupby('store_id')['day'].max()
    dsa = snapshot_day - topmap['store_id'].map(store_last)
    out['days_since_top_store_active'] = dsa
    out['top_store_active14'] = (dsa<=14).astype(float)
    # recent-week share of 28d spend
    s7 = tx[tx.day > snapshot_day-7].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    s28 = tx[tx.day > snapshot_day-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    out['fw_recent7_share'] = s7/(s28+1.0)
    return out

tab = agent_api.build_features(make_feats)
print('built:', tab.shape)
newcols = [c for c in tab.columns if c not in ('household_key','snapshot_day')]
print(tab[newcols].describe().round(3).T[['mean','std','min','max']])

e11 = agent_api.load_saved('e011_price.parquet')
m = e11.merge(tab, on=['household_key','snapshot_day'], how='inner')
print('after basestab merge:', m.shape)

nf = agent_api.load_saved('newfeat.parquet')
print('newfeat:', nf.shape, list(nf.columns))
ok = False
if nf.shape[0]==len(e11) and set(['household_key','snapshot_day']).issubset(nf.columns):
    nfc = [c for c in nf.columns if c not in ('household_key','snapshot_day')]
    if not set(nfc) & set(m.columns):
        m = m.merge(nf, on=['household_key','snapshot_day'], how='inner'); ok=True
print('after newfeat merge:', m.shape, 'merged_newfeat:', ok)
print('dups:', int(m.duplicated(['household_key','snapshot_day']).sum()))

tt = agent_api.train_targets()
mm = m.merge(tt, on=['household_key','snapshot_day'])
tr = mm.snapshot_day<=403
for c in newcols:
    print('corr', c, round(float(mm.loc[tr,c].corr(mm.loc[tr,'future_spend_4w'])),3))
path = agent_api.save_table(m, 'e018_basestab.parquet')
print('PATH:', path)
