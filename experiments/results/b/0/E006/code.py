
import numpy as np, pandas as pd

for name in ['mkt_demo','rfm28','rich_behavioral','season']:
    df = agent_api.load_saved(name + '.parquet')
    print('==', name, df.shape)
    print(list(df.columns))
    print()

tt = agent_api.train_targets()
y = tt.future_spend_4w
print('train rows', len(tt), 'mean', round(y.mean(),2), 'median', y.median(), 'zero share', round((y==0).mean(),3))
print('MAE median-pred', round((y - y.median()).abs().mean(),3), 'MAE mean-pred', round((y - y.mean()).abs().mean(),3))

rfm = agent_api.load_saved('rfm28.parquet')
m = tt.merge(rfm, on=['household_key','snapshot_day'], how='left')
for c in rfm.columns:
    if c not in ('household_key','snapshot_day'):
        print('MAE', c, '=', round((y - m[c]).abs().mean(),3))

base = agent_api.load_saved('mkt_demo.parquet')
mm = tt.merge(base, on=['household_key','snapshot_day'], how='left')
num = [c for c in base.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(mm[c])]
corrs = mm[num].corrwith(mm.future_spend_4w).sort_values()
print('\nTOP |corr| with target (mkt_demo):')
print(corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(15))

v = agent_api.snapshot()
print('\ntx rows<=459:', v.transactions.shape)
print('n departments:', v.products.department.nunique())
print('n commodities:', v.products.commodity_desc.nunique())


# ---- cell ----

import numpy as np, pandas as pd

tt = agent_api.train_targets()
y = tt.future_spend_4w.values

def mae(pred):
    return round(np.mean(np.abs(y - pred)),3)

season = agent_api.load_saved('season.parquet')
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')

# raw predictors
print('own_ly_spend4w MAE:', mae(m.own_ly_spend4w.values))
print('own_ly2_spend4w MAE:', mae(m.own_ly2_spend4w.values))
print('season_lift MAE:', mae(m.season_lift.values))
print('spend56 MAE:', mae(m.spend56.values))
print('spend112 MAE:', mae(m.spend112.values))
print('blend .5*28+.5*56:', mae(0.5*m.spend28.values+0.5*m.spend56.values))
print('blend .33/33/33:', mae((m.spend28+m.spend56+m.spend112)/3))
print('blend 28+ly:', mae(0.7*m.spend28.values+0.3*m.own_ly_spend4w.values))
print('blend 56+ly:', mae(0.7*m.spend56.values+0.3*m.own_ly_spend4w.values))
print('blend 112+ly:', mae(0.7*m.spend112.values+0.3*m.own_ly_spend4w.values))

# target stats by snapshot day (train)
g = tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median',lambda s:(s==0).mean()])
print('\nby snapshot day:\n', g.round(2))

# zero/nonzero split: does spend28 predict zeros?
z = m.future_spend_4w==0
print('\nzero rows: spend28 mean', round(m.spend28[z].mean(),2), ' nonzero rows:', round(m.spend28[~z].mean(),2))
print('spend28==0 share among zeros:', round((m.spend28[z]==0).mean(),3))
print('P(spend28==0) -> target zero:', round(m[m.spend28==0].future_spend_4w.eq(0).mean(),3))

# quick numpy ridge on season table (numeric only) to gauge linear signal
num = [c for c in season.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
X = m[num].fillna(0).values.astype(float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
mu, sd = X.mean(0), X.std(0)+1e-9
Xs = (X-mu)/sd
# add log1p versions of spend features
sp_cols = [i for i,c in enumerate(num) if c.startswith('spend') or c in ('lt_spend','own_ly_spend4w','own_ly2_spend4w')]
Xl = np.hstack([Xs, np.log1p(np.clip(X[:,sp_cols],0,None))])
def ridge(Xt, yt, lam=100.0):
    d = Xt.shape[1]
    A = Xt.T@Xt + lam*np.eye(d)
    return np.linalg.solve(A, Xt.T@yt)
w = ridge(Xl, y)
pred = Xl@w
print('\nridge(train-fit) MAE on train:', mae(pred))
w2 = ridge(Xl, np.log1p(y)); pred2 = np.expm1(Xl@w2)
print('ridge log-target MAE on train:', mae(pred2))


# ---- cell ----
import numpy as np, pandas as pd

season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
print('NaN counts (nonzero only):')
nc = m.isna().sum()
print(nc[nc>0])

y = m.future_spend_4w.values
def mae(pred):
    p = np.nan_to_num(np.asarray(pred,float), nan=0.0)
    return round(np.mean(np.abs(y-p)),3)
print('\nfilled MAE own_ly:', mae(m.own_ly_spend4w), ' own_ly2:', mae(m.own_ly2_spend4w), ' season_lift:', mae(m.season_lift))
print('filled MAE spend56:', mae(m.spend56), ' spend112:', mae(m.spend112))

# local eval harness: ridge fit on snapshots 95..403, eval on 431 (pseudo-validation)
def local_eval(df, lam=50.0, target='raw', eval_days=(403,431), verbose=True):
    num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(float)
    # log1p the heavy-tailed spend-like columns
    logc = [i for i,c in enumerate(num) if ('spend' in c or 'disc' in c or 'lt_' in c or 'own_ly' in c)]
    X = np.hstack([X, np.log1p(np.clip(X[:,logc],0,None))]) if logc else X
    res = {}
    for ed in eval_days:
        tr = df.snapshot_day < ed
        te = df.snapshot_day == ed
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xs = (X-mu)/sd
        yt = df.future_spend_4w.values
        if target=='log': yt = np.log1p(yt)
        A = Xs[tr].T@Xs[tr] + lam*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T@yt[tr])
        p = Xs[te]@w
        if target=='log': p = np.expm1(p)
        res[ed] = round(np.mean(np.abs(yt[te]-p)),3)
    if verbose: print('local ridge MAE', res, 'n_feat', X.shape[1])
    return res

r = local_eval(m)
# baselines on 431
te = m.snapshot_day==431; yte = m.future_spend_4w[te].values
for c in ['spend28','spend56','own_ly_spend4w','own_ly2_spend4w','season_lift']:
    print('431 MAE', c, round(np.mean(np.abs(yte - m[c][te].fillna(0).values)),3))
print('431 MAE blend .5/.5:', round(np.mean(np.abs(yte - (0.5*m.spend28[te]+0.5*m.spend56[te]).fillna(0).values)),3))
print('431 median pred:', round(np.mean(np.abs(yte - np.median(m.future_spend_4w[m.snapshot_day<431]))),3))

v = agent_api.snapshot()
dep = v.products.department.value_counts()
print('\ndepartments:', len(dep))
print(dep.head(20))

# ---- cell ----
import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w
for c in ['spend28','spend56','spend112','own_ly_spend4w','own_ly2_spend4w','season_lift','lt_spend','spend364']:
    s = m[c]
    print(f'{c:18s} mean {s.mean():9.1f} med {s.median():8.1f} p95 {s.quantile(.95):9.1f} max {s.max():12.1f} corr {s.corr(y):.3f}')
print()
print('spend28 describe:\n', m.spend28.describe())
print('spend56 describe:\n', m.spend56.describe())
# ratio check
r = (m.spend56/m.spend28.replace(0,np.nan)).dropna()
print('spend56/spend28 ratio quantiles:', r.quantile([.05,.25,.5,.75,.95]).round(2).values)
print('target quantiles:', y.quantile([.5,.9,.99]).values, 'max', y.max())

# ---- cell ----
import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values

# which snapshot days have own_ly_spend4w non-NaN?
print('own_ly non-NaN count by snapshot day:')
print(m.groupby('snapshot_day').own_ly_spend4w.agg(['count','mean']).round(1))

# pseudo-validation: ridge on 95..403, eval 431, using season table (no own_ly)
def local_eval(df, lam=50.0, target='raw', eval_days=(403,431), logpat=('spend','disc','lt_','own_ly')):
    num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(float)
    logc = [i for i,c in enumerate(num) if any(p in c for p in logpat)]
    if logc: X = np.hstack([X, np.log1p(np.clip(X[:,logc],0,None))])
    res = {}
    for ed in eval_days:
        tr = df.snapshot_day < ed; te = df.snapshot_day == ed
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xs = (X-mu)/sd
        yt = df.future_spend_4w.values
        if target=='log': yt = np.log1p(yt)
        A = Xs[tr].T@Xs[tr] + lam*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T@yt[tr])
        p = Xs[te]@w
        if target=='log': p = np.expm1(p)
        res[ed] = round(float(np.mean(np.abs(yt[te]-p))),3)
    return res, X.shape[1]

r, nf = local_eval(m)
print('season table local ridge:', r, 'nfeat', nf)
rich = agent_api.load_saved('rich_behavioral.parquet')
mr = tt.merge(rich, on=['household_key','snapshot_day'], how='left')
r2, nf2 = local_eval(mr)
print('rich_behavioral local ridge:', r2, 'nfeat', nf2)
# raw spend28-only "model": local
for lam in [10,50,200]:
    X = m[['spend28']].values.astype(float)
    tr = m.snapshot_day<431; te = m.snapshot_day==431
    mu,sd = X[tr].mean(0), X[tr].std(0)+1e-9
    Xs=(X-mu)/sd
    A=Xs[tr].T@Xs[tr]+lam*np.eye(1); w=np.linalg.solve(A,Xs[tr].T@y[tr])
    print('ridge spend28 lam',lam,'431 MAE', round(float(np.mean(np.abs(y[te]-Xs[te]*w))),3))

# ---- cell ----
import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def local_eval(df, lam=50.0, target='raw', eval_days=(403,431), logpat=('spend','disc','lt_','own_ly'), add_log=True):
    num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(float)
    logc = [i for i,c in enumerate(num) if any(p in c for p in logpat)]
    if add_log and logc: X = np.hstack([X, np.log1p(np.clip(X[:,logc],0,None))])
    res = {}
    for ed in eval_days:
        tr = df.snapshot_day < ed; te = df.snapshot_day == ed
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xs = (X-mu)/sd
        yt = y if target=='raw' else np.log1p(y)
        # center y with train mean -> intercept
        ymu = yt[tr].mean()
        A = Xs[tr].T@Xs[tr] + lam*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T@(yt[tr]-ymu))
        p = Xs[te]@w + ymu
        if target=='log': p = np.expm1(np.clip(p,0,8))
        res[ed] = round(float(np.mean(np.abs(y[te]-p))),3)
    return res, X.shape[1]

print('season w/ intercept:', local_eval(m))
print('season log-target  :', local_eval(m, target='log'))
print('season raw no-logX :', local_eval(m, add_log=False))
print('spend28 only       :', local_eval(m[['household_key','snapshot_day','spend28','future_spend_4w']], lam=10))
print('spend28+56+112     :', local_eval(m[['household_key','snapshot_day','spend28','spend56','spend112','future_spend_4w']], lam=10))
print('spend28 log-target :', local_eval(m[['household_key','snapshot_day','spend28','future_spend_4w']], lam=10, target='log'))
print('spend28+56 log-tgt :', local_eval(m[['household_key','snapshot_day','spend28','spend56','future_spend_4w']], lam=10, target='log'))

# power-law single feature: fit target = a*spend28^b via log-log on train(<431)
tr = m.snapshot_day<431; te = m.snapshot_day==431
mask = m.spend28[tr]>0
b, la = np.polyfit(np.log(m.spend28[tr][mask]), np.log1p(y[tr][mask]), 1)
p = np.expm1(la + b*np.log(np.clip(m.spend28[te],1e-9,None)))
print('power-law spend28 431 MAE:', round(float(np.mean(np.abs(y[te]-p))),3), 'b=',round(b,3))

# ratio target/spend28 by bucket
r = y/np.clip(m.spend28,1,None)
bins = pd.qcut(m.spend28, 8, duplicates='drop')
print('\nmedian ratio by spend28 bucket:')
print(m.groupby(bins, observed=True).apply(lambda g: pd.Series({'med_ratio': np.median(g.future_spend_4w/np.clip(g.spend28,1,None)), 'n':len(g)}), include_groups=False).round(2))

# ---- cell ----
import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def local_eval(df, lam=50.0, target='raw', eval_days=(403,431), logpat=('spend','disc','lt_','own_ly','red'), add_log=True):
    num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(float)
    logc = [i for i,c in enumerate(num) if any(p in c for p in logpat)]
    if add_log and logc: X = np.hstack([X, np.log1p(np.clip(X[:,logc],0,None))])
    res = {}
    for ed in eval_days:
        tr = df.snapshot_day < ed; te = df.snapshot_day == ed
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xs = (X-mu)/sd
        yt = y if target=='raw' else np.log1p(y)
        ymu = yt[tr].mean()
        A = Xs[tr].T@Xs[tr] + lam*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T@(yt[tr]-ymu))
        p = Xs[te]@w + ymu
        if target=='log': p = np.expm1(np.clip(p,0,8))
        res[ed] = round(float(np.mean(np.abs(y[te]-p))),3)
    return res, X.shape[1]

subs = {
 'core_spend': ['spend28','spend56','spend112','spend364','lt_spend'],
 'rfm_all':    [c for c in m.columns if c.startswith(('spend','trips','actdays','qty','nprod','nstore'))],
 'trends':     [c for c in m.columns if c.startswith('trend')],
 'disc':       ['coupon_disc112','retail_disc112','coupon_match_disc112','disc_share112'],
 'timing':     ['recency','tenure','mean_hour112','mean_dow112'],
 'season':     ['own_ly_spend4w','season_lift','week_sin','week_cos'],
 'demo':       [c for c in m.columns if c.startswith('demo_')],
 'mkt':        [c for c in m.columns if c.startswith(('n_tgt','tA','tB','tC','days_since','targeted','red'))],
}
for name, cols in subs.items():
    cols = [c for c in cols if c in m.columns]
    r, nf = local_eval(m[['household_key','snapshot_day']+cols], lam=20)
    print(f'{name:12s}', r, nf)

# ---- cell ----
import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def prep(df, logpat=('spend','disc','lt_','own_ly','red')):
    num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(float)
    logc = [i for i,c in enumerate(num) if any(p in c for p in logpat)]
    if logc: X = np.hstack([X, np.log1p(np.clip(X[:,logc],0,None))])
    return X

def local_eval(df, lam=50.0, eval_days=(403,431)):
    X = prep(df)
    res = {}
    for ed in eval_days:
        tr = df.snapshot_day < ed; te = df.snapshot_day == ed
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xs = (X-mu)/sd
        ymu = y[tr].mean()
        A = Xs[tr].T@Xs[tr] + lam*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T@(y[tr]-ymu))
        p = Xs[te]@w + ymu
        res[ed] = round(float(np.mean(np.abs(y[te]-p))),3)
    return res, X.shape[1]

# simple GBM via binned gradient boosting with depth-2 trees (numpy implementation)
def gbm_fit(X, yy, n_trees=300, lr=0.08, max_depth=3, min_leaf=20, subsample=0.7, seed=0):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    F = np.zeros(n)
    trees = []
    for t in range(n_trees):
        resid = yy - F
        idx = rng.choice(n, size=int(n*subsample), replace=False) if subsample<1 else np.arange(n)
        # build tree on resid[idx]
        def build(I, depth):
            if depth>=max_depth or len(I)<2*min_leaf:
                return ('leaf', resid[I].mean())
            best=(None,None,-1)
            for j in range(d):
                v = X[I,j]
                o = np.argsort(v, kind='stable')
                vs = v[o]; rs = resid[I][o]
                cs = np.cumsum(rs); tc = cs[-1]
                # candidate splits at unique boundaries
                valid = vs[1:] != vs[:-1]
                pos = np.where(valid)[0]+1
                if len(pos)==0: continue
                # evaluate every split but subsample candidate positions for speed
                if len(pos)>40:
                    pos = np.unique(np.linspace(0,len(pos)-1,40).astype(int))
                    pos = np.where(valid)[0][pos]+1
                gl = cs[pos-1]; gr = tc - gl
                nl = pos; nr = len(I)-pos
                gain = gl*gl/nl + gr*gr/nr
                k = np.argmax(gain)
                if gain[k] > best[2]:
                    best = (j, vs[pos[k]-1], gain[k])
            if best[0] is None: return ('leaf', resid[I].mean())
            j, thr, g = best
            L = I[X[I,j] <= thr]; R = I[X[I,j] > thr]
            if len(L)<min_leaf or len(R)<min_leaf: return ('leaf', resid[I].mean())
            return ('node', j, thr, build(L, depth+1), build(R, depth+1))
        tree = build(idx, 0)
        trees.append(tree)
        # update F on all rows
        def pred_one(T, x):
            while T[0]=='node':
                T = T[3] if x[T[1]] <= T[2] else T[4]
            return T[1]
        # vectorized predict for all n rows
        def predict(T):
            out = np.zeros(n)
            def rec(T, mask):
                if T[0]=='leaf':
                    out[mask] = T[1]; return
                v = X[mask, T[1]]
                rec(T[3], mask[v<=T[2]]); rec(T[4], mask[v>T[2]])
            rec(T, np.ones(n, bool))
            return out
        F += lr * predict(tree)
    return trees

def gbm_predict(trees, X, lr=0.08):
    n = X.shape[0]; out = np.zeros(n)
    for T in trees:
        def rec(T, mask):
            if T[0]=='leaf':
                out[mask] += T[1]*lr; return
            v = X[mask, T[1]]
            rec(T[3], mask[v<=T[2]]); rec(T[4], mask[v>T[2]])
        rec(T, np.ones(n, bool))
    return out

# test on 431: features = spend28 log, spend56 log, spend112 log, trips28, actdays28
feats = ['spend28','spend56','spend112','trips28','actdays28','nprod28','recency','tenure']
X = np.log1p(np.clip(m[feats].fillna(0).values,0,None))
tr = m.snapshot_day<431; te = m.snapshot_day==431
trees = gbm_fit(X[tr], y[tr], n_trees=200, lr=0.08)
p = gbm_predict(trees, X[te])
print('toy GBM 431 MAE:', round(float(np.mean(np.abs(y[te]-p))),3))
# compare ridge same feats
mu,sd = X[tr].mean(0), X[tr].std(0)+1e-9
Xs=(X-mu)/sd; A=Xs[tr].T@Xs[tr]+20*np.eye(len(feats)); w=np.linalg.solve(A,Xs[tr].T@(y[tr]-y[tr].mean()))
pr = Xs[te]@w + y[tr].mean()
print('ridge same feats 431 MAE:', round(float(np.mean(np.abs(y[te]-pr))),3))

# ---- cell ----
import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def make_bins(Xtr, nbin=64):
    edges = []
    for j in range(Xtr.shape[1]):
        qs = np.quantile(Xtr[:,j], np.linspace(0,1,nbin+1)[1:-1])
        edges.append(np.unique(qs))
    def binn(X):
        B = np.zeros(X.shape, dtype=np.int16)
        for j,e in enumerate(edges):
            B[:,j] = np.searchsorted(e, X[:,j], side='right')
        return B
    return binn

def fit_gbm(B, yy, n_trees=400, lr=0.06, max_depth=4, min_leaf=30, rowsample=0.7, colsample=0.8, seed=1):
    rng = np.random.RandomState(seed)
    n, d = B.shape
    F = np.zeros(n); trees = []
    for t in range(n_trees):
        resid = yy - F
        idx = rng.choice(n, int(n*rowsample), replace=False)
        cols = rng.choice(d, int(d*colsample), replace=False)
        tree = build(idx, cols, 0)
        trees.append((tree, cols))
        F += lr * pred_tree(tree, B, cols, np.arange(n))
    return trees

def build(idx, cols, depth, max_depth=4, min_leaf=30):
    if depth>=max_depth or len(idx)<2*min_leaf:
        return (-1, resid[idx].mean())
    tot = resid[idx].sum(); cnt = len(idx)
    base = tot*tot/cnt
    best = None
    for j in cols:
        b = B[idx, j]
        o = np.argsort(b, kind='stable')
        bs = b[o]; rs = resid[idx][o]
        cs = np.cumsum(rs)
        valid = np.zeros(len(idx), bool); valid[1:] = bs[1:]!=bs[:-1]
        pos = np.where(valid)[0]
        if len(pos)==0: continue
        gl = cs[pos]; gr = tot - gl
        nl = (pos+1).astype(float); nr = cnt - nl
        gain = gl*gl/np.maximum(nl,1) + gr*gr/np.maximum(nr,1) - base
        k = int(np.argmax(gain))
        if gain[k] <= 1e-9: continue
        if best is None or gain[k] > best[0]:
            best = (gain[k], j, bs[pos[k]])
    if best is None:
        return (-1, resid[idx].mean())
    g, j, thr = best
    mask = B[idx, j] <= thr
    L = idx[mask]; R = idx[~mask]
    if len(L)<min_leaf or len(R)<min_leaf:
        return (-1, resid[idx].mean())
    return (j, thr, build(L, cols, depth+1), build(R, cols, depth+1))

def pred_tree(tree, B, idx):
    j, thr, Ls, Rs = tree
    if j == -1:
        return np.full(len(idx), thr)
    mask = B[idx, j] <= thr
    out = np.zeros(len(idx))
    out[mask] = pred_tree(Ls, B, idx[mask])
    out[~mask] = pred_tree(Rs, B, idx[~mask])
    return out

feats = ['spend28','spend56','spend112','trips28','actdays28','nprod28','recency','tenure','trips56','nprod56']
X = np.log1p(np.clip(m[feats].fillna(0).values,0,None))
tr = m.snapshot_day<431; te = m.snapshot_day==431
binn = make_bins(X[tr])
Btr, Bte = binn(X[tr]), binn(X[te])
trees = fit_gbm(Btr, y[tr])
p = np.zeros(int(te.sum()))
for tree, cols in trees:
    p += 0.06 * pred_tree(tree, Bte, np.arange(len(p)))
print('toy GBM 431 MAE:', round(float(np.mean(np.abs(y[te]-p))),3))
mu,sd = X[tr].mean(0), X[tr].std(0)+1e-9
Xs=(X-mu)/sd; A=Xs[tr].T@Xs[tr]+20*np.eye(len(feats)); w=np.linalg.solve(A,Xs[tr].T@(y[tr]-y[tr].mean()))
pr = Xs[te]@w + y[tr].mean()
print('ridge same feats 431 MAE:', round(float(np.mean(np.abs(y[te]-pr))),3))

# ---- cell ----
import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def fit_gbm(B, yy, n_trees=400, lr=0.06, max_depth=4, min_leaf=30, rowsample=0.7, colsample=0.8, seed=1):
    rng = np.random.RandomState(seed)
    n, d = B.shape
    F = np.zeros(n); trees = []
    def build(idx, cols, depth):
        if depth>=max_depth or len(idx)<2*min_leaf:
            return (-1, resid[idx].mean())
        tot = resid[idx].sum(); cnt = len(idx)
        base = tot*tot/cnt
        best = None
        for j in cols:
            b = B[idx, j]
            o = np.argsort(b, kind='stable')
            bs = b[o]; rs = resid[idx][o]
            cs = np.cumsum(rs)
            valid = np.zeros(len(idx), bool); valid[1:] = bs[1:]!=bs[:-1]
            pos = np.where(valid)[0]
            if len(pos)==0: continue
            gl = cs[pos]; gr = tot - gl
            nl = (pos+1).astype(float); nr = cnt - nl
            gain = gl*gl/np.maximum(nl,1) + gr*gr/np.maximum(nr,1) - base
            k = int(np.argmax(gain))
            if gain[k] <= 1e-9: continue
            if best is None or gain[k] > best[0]:
                best = (gain[k], j, bs[pos[k]])
        if best is None:
            return (-1, resid[idx].mean())
        g, j, thr = best
        mask = B[idx, j] <= thr
        L = idx[mask]; R = idx[~mask]
        if len(L)<min_leaf or len(R)<min_leaf:
            return (-1, resid[idx].mean())
        return (j, thr, build(L, cols, depth+1), build(R, cols, depth+1))
    def pred_tree(tree, Bq, idx):
        j, thr, Ls, Rs = tree
        if j == -1:
            return np.full(len(idx), thr)
        mask = Bq[idx, j] <= thr
        out = np.zeros(len(idx))
        out[mask] = pred_tree(Ls, Bq, idx[mask])
        out[~mask] = pred_tree(Rs, Bq, idx[~mask])
        return out
    for t in range(n_trees):
        resid = yy - F
        idx = rng.choice(n, int(n*rowsample), replace=False)
        cols = rng.choice(d, int(d*colsample), replace=False)
        tree = build(idx, cols, 0)
        trees.append((tree, cols))
        F += lr * pred_tree(tree, B, np.arange(n))
    return trees

def gbm_pred(trees, B, lr=0.06):
    n = B.shape[0]; p = np.zeros(n)
    def pred_tree(tree, idx):
        j, thr, Ls, Rs = tree
        if j == -1:
            p[idx] += thr*lr; return
        mask = B[idx, j] <= thr
        pred_tree(Ls, idx[mask]); pred_tree(Rs, idx[~mask])
    for tree, cols in trees:
        pred_tree(tree, np.arange(n))
    return p

feats = ['spend28','spend56','spend112','trips28','actdays28','nprod28','recency','tenure','trips56','nprod56']
X = np.log1p(np.clip(m[feats].fillna(0).values,0,None))
tr = m.snapshot_day<431; te = m.snapshot_day==431
def make_bins(Xtr, nbin=64):
    edges = [np.unique(np.quantile(Xtr[:,j], np.linspace(0,1,nbin+1)[1:-1])) for j in range(Xtr.shape[1])]
    def binn(X):
        B = np.zeros(X.shape, dtype=np.int16)
        for j,e in enumerate(edges): B[:,j] = np.searchsorted(e, X[:,j], side='right')
        return B
    return binn
binn = make_bins(X[tr])
Btr, Bte = binn(X[tr]), binn(X[te])
trees = fit_gbm(Btr, y[tr])
p = gbm_pred(trees, Bte)
print('toy GBM 431 MAE:', round(float(np.mean(np.abs(y[te]-p))),3))
mu,sd = X[tr].mean(0), X[tr].std(0)+1e-9
Xs=(X-mu)/sd; A=Xs[tr].T@Xs[tr]+20*np.eye(len(feats)); w=np.linalg.solve(A,Xs[tr].T@(y[tr]-y[tr].mean()))
pr = Xs[te]@w + y[tr].mean()
print('ridge same feats 431 MAE:', round(float(np.mean(np.abs(y[te]-pr))),3))

# ---- cell ----
import numpy as np, pandas as pd

# Test 1: can build_features fn use load_saved? trivial test on 2 snapshots via manual call pattern
saved = agent_api.load_saved('mkt_demo.parquet')

def test_fn(view, snapshot_day):
    hh = view.households
    sub = saved[saved.snapshot_day == snapshot_day][['household_key','spend28']]
    return hh[['household_key']].merge(sub, on='household_key', how='left').set_index('household_key')

# don't run full build_features (expensive); just check one snapshot manually
v = agent_api.snapshot(431)
out = test_fn(v, 431)
print('merge-in-fn OK:', out.shape, out.spend28.notna().mean())

# Test 2: time + sanity of new features at snapshot 431
v = agent_api.snapshot(431)
tx = v.transactions
prods = v.products[['product_id','department']]
tx = tx.merge(prods, on='product_id', how='left')
d0 = 431
tx28 = tx[tx.day > d0-28]
print('tx28 rows:', len(tx28))

# dept spend pivot
ds = tx28.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
ds.columns = ['dsp_' + c for c in ds.columns]
print('dept spend cols:', ds.shape)

# weekly spend pattern last 28d
w = tx28.assign(wk=(d0 - tx28.day)//7)  # 0=most recent week
ws = w.groupby(['household_key','wk']).sales_value.sum().unstack(fill_value=0.0)
ws.columns = [f'spend_w{c}' for c in ws.columns]
print('weekly cols:', list(ws.columns))

# trip intervals last 112d
tx112 = tx[tx.day > d0-112]
g = tx112.groupby('household_key').day.apply(lambda s: np.diff(np.sort(s.unique())))
iv = pd.DataFrame({'hh': g.index, 'iv': g.values})
iv['mean_iv'] = iv.iv.apply(lambda a: a.mean() if len(a) else np.nan)
iv['std_iv'] = iv.iv.apply(lambda a: a.std() if len(a)>1 else 0.0)
iv['min_iv'] = iv.iv.apply(lambda a: a.min() if len(a) else np.nan)
print('interval stats sample:', iv[['mean_iv','std_iv','min_iv']].describe().round(1))

# basket stats 28d
bk = tx28.groupby(['household_key','basket_id']).sales_value.sum()
bs = bk.groupby('household_key').agg(bk_mean28='mean', bk_max28='max', bk_std28='std', nbaskets28='count')
print(bs.describe().round(1))

# recency-weighted spend 84d
tx84 = tx[tx.day > d0-84]
rw = tx84.assign(w=np.exp(-(d0-tx84.day)/28.0)).groupby('household_key').apply(lambda g: (g.sales_value*g.w).sum())
print('recency-weighted spend:', rw.describe().round(1))

# ---- cell ----
import numpy as np, pandas as pd

def fn(view, snapshot_day):
    d0 = int(snapshot_day)
    hh_df = view.households
    try:
        k = hh_df['household_key']
    except Exception:
        k = hh_df
    keys = pd.Index(pd.unique(np.asarray(k).ravel()), name='household_key')
    tx = view.transactions
    out = {}
    def put(name, s, fill=np.nan):
        try:
            out[name] = s.reindex(keys).astype(float).fillna(fill)
        except Exception:
            out[name] = pd.Series(fill, index=keys, name=name)

    # multi-window RFM
    for w in (7, 28, 56, 112, 364):
        t = tx[tx.day > d0 - w]
        g = t.groupby('household_key')
        put(f'spend{w}', g.sales_value.sum(), 0.0)
        put(f'trips{w}', g.basket_id.nunique(), 0.0)
        put(f'actdays{w}', g.day.nunique(), 0.0)
        put(f'qty{w}', g.quantity.sum(), 0.0)
        put(f'nprod{w}', g.product_id.nunique(), 0.0)
        put(f'nstore{w}', g.store_id.nunique(), 0.0)
    g = tx.groupby('household_key')
    put('recency', d0 - g.day.max(), 364.0)
    put('tenure', d0 - g.day.min(), 0.0)
    put('lt_spend', g.sales_value.sum(), 0.0)
    put('lt_trips', g.basket_id.nunique(), 0.0)
    s7, s28, s56, s112, s364 = out['spend7'], out['spend28'], out['spend56'], out['spend112'], out['spend364']
    put('trend_7_28', 4*s7/(s28+1.0), 0.0)
    put('trend_28_56', 2*s28/(s56+1.0), 0.0)
    put('trend_56_112', 2*s56/(s112+1.0), 0.0)
    put('trend_112_364', (364/112)*s112/(s364+1.0), 0.0)
    put('avg_basket28', s28/out['trips28'].replace(0, np.nan), 0.0)
    put('spend_per_day28', s28/28.0, 0.0)
    put('trips_per_day28', out['trips28']/28.0, 0.0)
    if d0 - 363 >= 1:
        t = tx[(tx.day >= d0-363) & (tx.day <= d0-336)]
        put('spend_yoy28', t.groupby('household_key').sales_value.sum(), 0.0)
    else:
        out['spend_yoy28'] = pd.Series(np.nan, index=keys)
    t = tx[tx.day > d0-112]
    gt = t.groupby('household_key')
    c1, c2, c3 = gt.coupon_disc.sum(), gt.coupon_match_disc.sum(), gt.retail_disc.sum()
    put('coupon_disc112', c1, 0.0); put('coupon_match_disc112', c2, 0.0); put('retail_disc112', c3, 0.0)
    put('disc_share112', (-(c1+c2+c3))/(out['spend112']+1.0), 0.0)
    put('mean_hour112', (gt.trans_time.mean()//100), 12.0)
    put('mean_dow112', gt.day.apply(lambda s: s.mod(7).mean()), 3.0)

    # dept merge
    txm = tx.merge(view.products[['product_id','department']], on='product_id', how='left')
    put('ndept112', txm[txm.day > d0-112].groupby('household_key').department.nunique(), 0.0)

    # calendar / season
    wk = (d0 + 8)//7
    out['week'] = pd.Series(float(wk), index=keys)
    out['week_sin'] = pd.Series(float(np.sin(2*np.pi*wk/52)), index=keys)
    out['week_cos'] = pd.Series(float(np.cos(2*np.pi*wk/52)), index=keys)
    try:
        wksp = tx.groupby('week_no').sales_value.sum()
        wksp = wksp[wksp.index <= wk]
        mw = wksp.mean()
        lifts = [wksp[wk-52*k2]/mw for k2 in (1,) for wk in [wk+k2] if 1 <= wk-52 <= wk and (wk-52) in wksp.index]
        lifts = []
        for k2 in (1, 2, 3, 4):
            w2 = wk + k2; ref = w2 - 52
            if 1 <= ref <= wk and ref in wksp.index:
                lifts.append(wksp[ref]/mw)
        out['season_lift'] = pd.Series(float(np.mean(lifts)) if lifts else np.nan, index=keys)
    except Exception:
        out['season_lift'] = pd.Series(np.nan, index=keys)
    for nm, (a, b) in {'own_ly_spend4w': (d0-363, d0-336), 'own_ly2_spend4w': (d0-727, d0-700)}.items():
        if a >= 1:
            t = tx[(tx.day >= a) & (tx.day <= b)]
            put(nm, t.groupby('household_key').sales_value.sum(), 0.0)
        else:
            out[nm] = pd.Series(np.nan, index=keys)

    # marketing
    try:
        cps, tg = view.campaigns, view.campaign_targets
        red, cpns = view.coupon_redemptions, view.coupons
        ever_ids = set(cps.campaign[cps.start_day <= d0]); act_ids = set(cps.campaign[(cps.start_day <= d0) & (cps.end_day >= d0)])
        tge = tg[tg.campaign.isin(ever_ids)]; tga = tg[tg.campaign.isin(act_ids)]
        ne = tge.groupby('household_key').campaign.nunique(); na = tga.groupby('household_key').campaign.nunique()
        put('n_tgt_ever', ne, 0.0); put('n_tgt_active', na, 0.0)
        put('targeted_flag', (ne > 0).astype(float), 0.0)
        sd_map = cps.set_index('campaign').start_day; ed_map = cps.set_index('campaign').end_day
        tge2 = tge.assign(sd=tge.campaign.map(sd_map))
        put('days_since_first_tgt', d0 - tge2.groupby('household_key').sd.min())
        put('days_since_last_tgt', d0 - tge2.groupby('household_key').sd.max())
        tga2 = tga.assign(rem=(tga.campaign.map(ed_map) - d0).clip(lower=0))
        put('tgt_active_remaining', tga2.groupby('household_key').rem.max(), 0.0)
        desc = tge.description.astype(str).str.upper()
        for L in ('A', 'B', 'C'):
            put(f't{L}_lt', tge[desc.str.endswith(L)].groupby('household_key').size().gt(0).astype(float), 0.0)
            desca = tga.description.astype(str).str.upper()
            put(f't{L}_act', tga[desca.str.endswith(L)].groupby('household_key').size().gt(0).astype(float), 0.0)
        r = red[red.day <= d0]
        put('red_lt', r.groupby('household_key').size(), 0.0)
        put('red28', r[r.day > d0-28].groupby('household_key').size(), 0.0)
        put('red112', r[r.day > d0-112].groupby('household_key').size(), 0.0)
        put('red_ncamp', r.groupby('household_key').campaign.nunique(), 0.0)
        put('red_active', r[r.campaign.isin(act_ids)].groupby('household_key').size(), 0.0)
        put('days_since_last_red', d0 - r.groupby('household_key').day.max())
        put('red_rate', out['red_lt']/(out['lt_trips']+1.0), 0.0)
        rc = r[['household_key','campaign']].drop_duplicates()
        cp = cpns.merge(rc, on='campaign')[['household_key','product_id']].drop_duplicates()
        sp = tx[tx.day > d0-28].merge(cp, on=['household_key','product_id']).groupby('household_key').sales_value.sum()
        put('cpn_prod_spend28', sp, 0.0)
        put('cpn_prod_share28', sp/(s28+1.0), 0.0)
    except Exception:
        for c in ['n_tgt_ever','n_tgt_active','targeted_flag','days_since_first_tgt','days_since_last_tgt','tgt_active_remaining','tA_lt','tB_lt','tC_lt','tA_act','tB_act','tC_act','red_lt','red28','red112','red_ncamp','red_active','days_since_last_red','red_rate','cpn_prod_spend28','cpn_prod_share28']:
            out[c] = pd.Series(np.nan, index=keys)

    # demographics
    try:
        dm = view.demographics
        agem = dm.classification_1.astype(str).str.extract(r'(\d+)')[0].astype(float)
        l3m = dm.classification_3.astype(str).str.extract(r'(\d+)')[0].astype(float)
        hsm = dm.classification_4.astype(str).str.extract(r'(\d+)')[0].astype(float)
        kidm = dm.kid_category_desc.astype(str).map({'None/Unknown': 0.0, '1': 1.0, '2': 2.0, '3+': 3.0})
        homm = dm.homeowner_desc.astype(str).map({'Homeowner': 3.0, 'Probable Owner': 2.0, 'Probable Renter': 1.0, 'Renter': 0.0, 'Unknown': np.nan})
        c2m = dm.classification_2.astype(str).map({'X': 0.0, 'Y': 1.0, 'Z': 2.0})
        c5m = dm.classification_5.astype(str).str.extract(r'(\d+)')[0].astype(float)
        dmi = dm.set_index('household_key')
        put('demo_age', agem.set_axis(dmi.index)); put('demo_class3', l3m.set_axis(dmi.index))
        put('demo_hhsize', hsm.set_axis(dmi.index)); put('demo_kids', kidm.set_axis(dmi.index))
        put('demo_homeowner', homm.set_axis(dmi.index)); put('demo_c2', c2m.set_axis(dmi.index))
        put('demo_c5', c5m.set_axis(dmi.index))
        out['has_demo'] = dm.household_key.isin(keys).groupby(dm.household_key).max().reindex(keys).astype(float).fillna(0)
        out['has_demo'] = pd.Series(0.0, index=keys).where(~pd.Series(keys).isin(set(dm.household_key)), 1.0)
    except Exception:
        for c in ['demo_age','demo_class3','demo_hhsize','demo_kids','demo_homeowner','demo_c2','demo_c5','has_demo']:
            out[c] = pd.Series(np.nan, index=keys)

    # NEW: composition / cadence / tail features
    tx28 = tx[tx.day > d0-28]; txm28 = txm[txm.day > d0-28]
    try:
        top = ['GROCERY','DRUG GM','PRODUCE','COSMETICS','NUTRITION','MEAT','MEAT-PCKGD','DELI','PASTRY','FLORAL','SEAFOOD-PCKGD','MISC. TRANS.']
        dsp = txm28[txm28.department.isin(top)].groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
        dsp = dsp.reindex(keys).fillna(0.0)
        sh = dsp.div(s28.where(s28 > 0, np.nan), axis=0).fillna(0.0)
        for c in sh.columns: out['dsp_' + str(c)] = sh[c]
        put('ndept28', txm28.groupby('household_key').department.nunique(), 0.0)
    except Exception:
        pass
    try:
        wkn = ((d0 - tx28.day)//7).clip(0, 3)
        ws = tx28.groupby([tx28.household_key, wkn]).sales_value.sum().unstack(fill_value=0.0).reindex(keys).fillna(0.0)
        for c in ws.columns: out[f'spend_w{int(c)}'] = ws[c]
        out['share_w0'] = ws[0].astype(float)/(s28+1.0) if 0 in ws.columns else pd.Series(0.0, index=keys)
    except Exception:
        pass
    try:
        def ivstats(s):
            u = np.sort(s.unique()); dd = np.diff(u)
            return pd.Series({'iv_mean112': dd.mean() if len(dd) else np.nan,
                              'iv_std112': dd.std() if len(dd) > 1 else 0.0,
                              'iv_min112': dd.min() if len(dd) else np.nan,
                              'n_gaps112': float(len(dd))})
        ivs = tx[tx.day > d0-112].groupby('household_key').day.apply(ivstats).unstack()
        for c in ['iv_mean112','iv_std112','iv_min112','n_gaps112']: put(c, ivs[c])
    except Exception:
        pass
    try:
        bk = tx28.groupby(['household_key','basket_id']).sales_value.sum()
        bs = bk.groupby(level=0).agg(['mean','max','std','count'])
        put('bk_mean28', bs['mean'], 0.0); put('bk_max28', bs['max'], 0.0)
        put('bk_std28', bs['std'], 0.0); put('nbaskets28', bs['count'], 0.0)
    except Exception:
        pass
    try:
        t84 = tx[tx.day > d0-84]
        wgt = np.exp(-(d0 - t84.day)/28.0)
        put('rwspend84', (t84.sales_value*wgt).groupby(t84.household_key).sum(), 0.0)
        put('active_weeks112', tx[tx.day > d0-112].groupby('household_key').week_no.nunique(), 0.0)
        we = txm28.assign(we=(txm28.day % 7 >= 5).astype(float))
        put('weekend_share28', we[we.we > 0].groupby('household_key').sales_value.sum()/(s28+1.0), 0.0)
    except Exception:
        pass
    out['spend28_log'] = np.log1p(s28.clip(lower=0)); out['spend56_log'] = np.log1p(s56.clip(lower=0))
    out['spend112_log'] = np.log1p(s112.clip(lower=0)); out['lt_spend_log'] = np.log1p(out['lt_spend'].clip(lower=0))
    out['zero28'] = (s28 <= 0).astype(float)
    out['ratio28_364'] = s28/(s364+1.0)
    out['toptrip_share28'] = out.get('bk_max28', pd.Series(0.0, index=keys))/(s28+1.0)

    return pd.DataFrame(out)

path = agent_api.save_table(agent_api.build_features(fn), 'e006_composition')
print('saved:', path)
df = agent_api.load_saved('e006_composition.parquet') if agent_api.load_saved('e006_composition.parquet') is not None else None
print('shape check via build ok')