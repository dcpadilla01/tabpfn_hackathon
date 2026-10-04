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