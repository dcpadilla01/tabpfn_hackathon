
import agent_api, pandas as pd, numpy as np, time

t = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
train_days = agent_api.snapshot_days()['train']

# ---- numpy histogram GBM proxy ----
def fit_gbm(X, y, n_bins=64, n_trees=300, lr=0.05, max_depth=4, min_leaf=40):
    n, p = X.shape
    bins = np.zeros((p, n_bins+1))
    for j in range(p):
        qs = np.nanquantile(X[:,j], np.linspace(0,1,n_bins+1))
        qs = np.unique(qs)
        bins[j] = qs
    Xb = np.zeros((n,p), dtype=np.int16)
    for j in range(p):
        Xb[:,j] = np.searchsorted(bins[j][1:-1], X[:,j], side='right')
    F = np.zeros(n); trees=[]
    for it in range(n_trees):
        r = y - F
        node_idx = [np.arange(n)]
        tree=[]
        for d in range(max_depth):
            next_nodes=[]
            for idx in node_idx:
                if len(idx) < 2*min_leaf:
                    tree.append(('leaf', float(r[idx].mean()) if len(idx)>0 else 0.0)); next_nodes.append(None); continue
                best=(None,None,-1e18)
                rs = r[idx]
                tot = rs.sum(); tots = (rs*rs).sum()
                for j in range(p):
                    b = Xb[idx,j]
                    order = np.argsort(b, kind='stable')
                    bo = b[order]; ro = rs[order]
                    # cumulative sums at unique boundaries
                    csum = np.concatenate([[0], np.cumsum(ro)])
                    cs2 = np.concatenate([[0], np.cumsum(ro*ro)])
                    cnt = np.arange(1, len(idx)+1)
                    # valid split positions: where bo changes and both sides >= min_leaf
                    valid = np.zeros(len(idx), bool)
                    ch = bo[1:] != bo[:-1]
                    valid[1:] = ch
                    valid &= (cnt >= min_leaf) & (cnt <= len(idx)-min_leaf)
                    if not valid.any(): continue
                    left_s = csum[1:][valid]; left_c = cnt[1:][valid]
                    right_s = tot-left_s; right_c = len(idx)-left_c
                    # SSE reduction = (left_s^2/left_c + right_s^2/right_c) - tot^2/n
                    score = left_s*left_s/left_c + right_s*right_s/right_c
                    k = np.argmax(score)
                    pos = np.where(valid)[0][k]
                    thr = bo[pos]
                    gain = score[k] - tot*tot/len(idx)
                    if gain > best[2]:
                        best=(j, thr, gain)
                if best[0] is None or best[2] <= 0:
                    tree.append(('leaf', float(rs.mean()))); next_nodes.append(None); continue
                j, thr, g = best
                mask = Xb[idx,j] <= np.searchsorted(bins[j][1:-1], thr, side='right')-1
                mask = Xb[idx,j] <= thr if False else (Xb[idx,j] <= np.searchsorted(bins[j][1:-1], thr, side='left'))
                li = idx[mask]; ri = idx[~mask]
                tree.append(('split', j, thr))
                next_nodes.append(li); next_nodes.append(ri)
            node_idx = next_nodes
        # assign leaves: predict mean residual of each leaf
        # walk tree
        def pred_node(idx, node_i, out):
            node = tree[node_i]
            if node[0]=='leaf':
                if len(idx): out[idx] = node[1]
                return
            j, thr = node[1], node[2]
            mask = Xb[idx,j] <= np.searchsorted(bins[j][1:-1], thr, side='left')
            pred_node(idx[mask], 2*node_i+1, out)
            pred_node(idx[~mask], 2*node_i+2, out)
        upd = np.zeros(n)
        pred_node(np.arange(n), 0, upd)
        F += lr*upd
        trees.append((bins.copy(), tree))
    return F, trees

def gbm_predict(X, trees):
    n = X.shape[0]; F=np.zeros(n)
    for bins, tree in trees:
        Xb = np.zeros((n, X.shape[1]), dtype=np.int16)
        for j in range(X.shape[1]):
            Xb[:,j] = np.searchsorted(bins[j][1:-1], X[:,j], side='right')
        def pred_node(idx, node_i, out):
            node = tree[node_i]
            if node[0]=='leaf':
                if len(idx): out[idx]=node[1]
                return
            j, thr = node[1], node[2]
            mask = Xb[idx,j] <= np.searchsorted(bins[j][1:-1], thr, side='left')
            pred_node(idx[mask], 2*node_i+1, out); pred_node(idx[~mask], 2*node_i+2, out)
        pred_node(np.arange(n), 0, F)
    return F

def cv_proxy(df, feats, n_trees=250, lr=0.05, log_target=False):
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
print('E005 proxy MAE:', round(r,3), 'time', round(time.time()-t0,1))
