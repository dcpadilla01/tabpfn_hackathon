
import agent_api, pandas as pd, numpy as np, time

t = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
train_days = agent_api.snapshot_days()['train']

def fit_gbm(X, y, n_bins=48, n_trees=200, lr=0.06, max_depth=4, min_leaf=40):
    n, p = X.shape
    bins=[]
    for j in range(p):
        bins.append(np.unique(np.nanquantile(X[:,j], np.linspace(0,1,n_bins+1))))
    Xb = np.zeros((n,p), dtype=np.int16)
    for j in range(p):
        Xb[:,j] = np.searchsorted(bins[j][1:-1], X[:,j], side='right')
    F = np.zeros(n); trees=[]
    for it in range(n_trees):
        r = y - F
        def build(idx, depth):
            if len(idx)==0: return ('leaf', 0.0)
            if depth>=max_depth or len(idx)<2*min_leaf:
                return ('leaf', float(r[idx].mean()))
            rs = r[idx]; tot = rs.sum()
            best=(None,None,-1e18)
            for j in range(p):
                b = Xb[idx,j]
                order = np.argsort(b, kind='stable')
                bo = b[order]; ro = rs[order]
                csum = np.concatenate([[0.0], np.cumsum(ro)])
                cnt = np.arange(1, len(idx)+1)
                valid = np.zeros(len(idx), bool)
                valid[1:] = bo[1:] != bo[:-1]
                valid &= (cnt >= min_leaf) & (cnt <= len(idx)-min_leaf)
                if not valid.any(): continue
                left_s = csum[1:][valid]; left_c = cnt[valid]
                right_s = tot-left_s; right_c = len(idx)-left_c
                score = left_s*left_s/left_c + right_s*right_s/right_c
                k = int(np.argmax(score))
                pos = np.where(valid)[0][k]
                thr = bo[pos]
                gain = score[k] - tot*tot/len(idx)
                if gain > best[2]: best=(j, thr, gain)
            if best[0] is None or best[2] <= 1e-10:
                return ('leaf', float(rs.mean()))
            j, thr, g = best
            thr_bin = int(np.searchsorted(bins[j][1:-1], thr, side='left'))
            mask = Xb[idx,j] <= thr_bin
            return ('split', j, thr_bin, build(idx[mask], depth+1), build(idx[~mask], depth+1))
        tree = build(np.arange(n), 0)
        upd = np.zeros(n)
        def pred(node, idx):
            if node[0]=='leaf':
                if len(idx): upd[idx]=node[1]
                return
            mask = Xb[idx,node[1]] <= node[2]
            pred(node[3], idx[mask]); pred(node[4], idx[~mask])
        pred(tree, np.arange(n))
        F += lr*upd
        trees.append((bins, tree))
    return F, trees

def gbm_predict(X, trees):
    n = X.shape[0]; F=np.zeros(n)
    for bins, tree in trees:
        Xb = np.zeros((n, X.shape[1]), dtype=np.int16)
        for j in range(X.shape[1]):
            Xb[:,j] = np.searchsorted(bins[j][1:-1], X[:,j], side='right')
        def pred(node, idx):
            if node[0]=='leaf':
                if len(idx): F[idx]+=node[1]
                return
            mask = Xb[idx,node[1]] <= node[2]
            pred(node[3], idx[mask]); pred(node[4], idx[~mask])
        pred(tree, np.arange(n))
    return F

def cv_proxy(df, feats, n_trees=200, lr=0.06, log_target=False):
    y = df['future_spend_4w'].values.astype(float)
    yt = np.log1p(y) if log_target else y
    maes=[]
    for d in train_days:
        tr = (df['snapshot_day']!=d).values; va = (df['snapshot_day']==d).values
        Xtr = df.loc[tr, feats].values.astype(float); Xva = df.loc[va, feats].values.astype(float)
        F, trees = fit_gbm(Xtr, yt[tr], n_trees=n_trees, lr=lr)
        p = gbm_predict(Xva, trees)
        if log_target: p = np.expm1(p)
        p = np.clip(p, 0, None)
        maes.append(np.mean(np.abs(p - y[va])))
    return float(np.mean(maes))

FEATS = [c for c in t.columns if c not in ('household_key','snapshot_day')]
t0=time.time()
r = cv_proxy(m, FEATS)
print('E005 proxy MAE raw:', round(r,3), 'time', round(time.time()-t0,1))
