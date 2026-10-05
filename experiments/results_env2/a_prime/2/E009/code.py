
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e005_decay_gapcv.parquet')
print('shape', t.shape)
print('cols', t.columns.tolist())
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
y = m['future_spend_4w']
print(y.describe())
print('zero frac', (y==0).mean(), 'median', y.median())
num = [c for c in t.columns if c not in ('household_key','snapshot_day')]
rows=[]
for c in num:
    x = pd.to_numeric(m[c], errors='coerce')
    ok = x.notna() & y.notna()
    if ok.sum()>100:
        rows.append((c, np.corrcoef(x[ok], y[ok])[0,1]))
cs = pd.DataFrame(rows, columns=['feat','corr']).reindex(pd.Series([r[0] for r in rows])).assign(corr=[r[1] for r in rows]).set_index('feat')['corr']
cs = cs.reindex(cs.abs().sort_values(ascending=False).index)
print(cs.round(3).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w']
rows=[]
for c in t.columns:
    if c in ('household_key','snapshot_day'): continue
    x = pd.to_numeric(m[c], errors='coerce')
    ok = x.notna() & y.notna()
    if ok.sum()>100:
        rows.append((c, float(np.corrcoef(x[ok], y[ok])[0,1])))
cs = pd.DataFrame(rows, columns=['feat','corr']).set_index('feat')
cs['abs'] = cs['corr'].abs()
print(cs.sort_values('abs', ascending=False).round(3).to_string())
# log-target correlations too
yl = np.log1p(y)
rows=[]
for c in cs.index:
    x = pd.to_numeric(m[c], errors='coerce')
    ok = x.notna() & y.notna()
    rows.append((c, float(np.corrcoef(x[ok], yl[ok])[0,1])))
csl = pd.DataFrame(rows, columns=['feat','corr_log']).set_index('feat')
print(csl.reindex(cs.sort_values('abs',ascending=False).index).round(3).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
train_days = agent_api.snapshot_days()['train']

FEATS = [c for c in t.columns if c not in ('household_key','snapshot_day')]

def ridge_fit_pred(Xtr, ytr, Xva, lam=1.0):
    # standardize
    mu = np.nanmean(Xtr, axis=0); sd = np.nanstd(Xtr, axis=0); sd[sd==0]=1
    Ztr = (Xtr-mu)/sd; Zva = (Xva-mu)/sd
    Ztr = np.nan_to_num(Ztr, nan=0.0); Zva = np.nan_to_num(Zva, nan=0.0)
    # add intercept + snapshot one-hot handled outside
    A = Ztr.T@Ztr + lam*np.eye(Ztr.shape[1])
    b = Ztr.T@ytr
    w = np.linalg.solve(A, b)
    return Zva@w

def cv_mae(df, feats, target='future_spend_4w', log_target=False, lam=1.0, add_day_dummies=True):
    df = df.copy()
    y = df[target].values.astype(float)
    yt = np.log1p(y) if log_target else y
    maes=[]
    for d in train_days:
        tr = df['snapshot_day']!=d; va = df['snapshot_day']==d
        Xtr = df.loc[tr, feats].values.astype(float); Xva = df.loc[va, feats].values.astype(float)
        if add_day_dummies:
            D = pd.get_dummies(df['snapshot_day']).astype(float)
            Dtr = D.loc[tr].values; Dva = D.loc[va].values
            Xtr = np.hstack([Xtr, Dtr]); Xva = np.hstack([Xva, Dva])
        p = ridge_fit_pred(Xtr, yt[tr.values], Xva, lam)
        if log_target: p = np.expm1(p)
        p = np.clip(p, 0, None)
        maes.append(np.mean(np.abs(p - y[va.values])))
    return float(np.mean(maes))

base = cv_mae(m, FEATS)
base_log = cv_mae(m, FEATS, log_target=True)
print('E005 ridge proxy MAE raw-target:', round(base,3))
print('E005 ridge proxy MAE log-target:', round(base_log,3))
# simple baselines
for name, pred in [('global median', np.full(len(m), m['future_spend_4w'].median())),
                   ('ew28*1.0', m['ew_28'].values), ('spend28*1.0', m['spend_28'].values)]:
    print(name, round(float(np.mean(np.abs(pred - m['future_spend_4w']))),3))


# ---- cell ----

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


# ---- cell ----

import agent_api, pandas as pd, numpy as np, time

t = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
train_days = agent_api.snapshot_days()['train']

def fit_gbm(X, y, n_bins=64, n_trees=300, lr=0.05, max_depth=4, min_leaf=40):
    n, p = X.shape
    bins=[]
    for j in range(p):
        qs = np.unique(np.nanquantile(X[:,j], np.linspace(0,1,n_bins+1)))
        bins.append(qs)
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
                if idx is None:
                    tree.append(('leaf', 0.0)); next_nodes += [None,None]; continue
                if len(idx) < 2*min_leaf:
                    tree.append(('leaf', float(r[idx].mean()))); next_nodes += [None,None]; continue
                rs = r[idx]; tot = rs.sum()
                best=(None,None,-1e18)
                for j in range(p):
                    b = Xb[idx,j]
                    order = np.argsort(b, kind='stable')
                    bo = b[order]; ro = rs[order]
                    csum = np.concatenate([[0], np.cumsum(ro)])
                    cnt = np.arange(1, len(idx)+1)
                    valid = np.zeros(len(idx), bool)
                    valid[1:] = bo[1:] != bo[:-1]
                    valid &= (cnt >= min_leaf) & (cnt <= len(idx)-min_leaf)
                    if not valid.any(): continue
                    left_s = csum[1:][valid]; left_c = cnt[1:][valid]
                    right_s = tot-left_s; right_c = len(idx)-left_c
                    score = left_s*left_s/left_c + right_s*right_s/right_c
                    k = np.argmax(score)
                    pos = np.where(valid)[0][k]
                    thr = bo[pos]
                    gain = score[k] - tot*tot/len(idx)
                    if gain > best[2]:
                        best=(j, thr, gain)
                if best[0] is None or best[2] <= 0:
                    tree.append(('leaf', float(rs.mean()))); next_nodes += [None,None]; continue
                j, thr, g = best
                thr_bin = np.searchsorted(bins[j][1:-1], thr, side='left')
                mask = Xb[idx,j] <= thr_bin
                li = idx[mask]; ri = idx[~mask]
                tree.append(('split', j, thr_bin))
                next_nodes += [li, ri]
            node_idx = next_nodes
        upd = np.zeros(n)
        def pred_node(idx, node_i, out):
            node = tree[node_i]
            if node[0]=='leaf':
                if len(idx): out[idx]=node[1]
                return
            j, thr = node[1], node[2]
            mask = Xb[idx,j] <= thr
            pred_node(idx[mask], 2*node_i+1, out); pred_node(idx[~mask], 2*node_i+2, out)
        pred_node(np.arange(n), 0, upd)
        F += lr*upd
        trees.append((bins, tree))
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
            mask = Xb[idx,j] <= thr
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
print('E005 proxy MAE raw:', round(r,3), 'time', round(time.time()-t0,1))
t0=time.time()
rl = cv_proxy(m, FEATS, log_target=True)
print('E005 proxy MAE log:', round(rl,3), 'time', round(time.time()-t0,1))


# ---- cell ----

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


# ---- cell ----

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
                score = left_s*left_s/left_c + (tot-left_s)*(tot-left_s)/(len(idx)-left_c)
                k = int(np.argmax(score))
                pos = np.where(valid)[0][k]
                thr = int(bo[pos])
                gain = score[k] - tot*tot/len(idx)
                if gain > best[2]: best=(j, thr, gain)
            if best[0] is None or best[2] <= 1e-10:
                return ('leaf', float(rs.mean()))
            j, thr, g = best
            mask = Xb[idx,j] <= thr
            return ('split', j, thr, build(idx[mask], depth+1), build(idx[~mask], depth+1))
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
t0=time.time()
rl = cv_proxy(m, FEATS, log_target=True)
print('E005 proxy MAE log:', round(rl,3), 'time', round(time.time()-t0,1))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot()
tx = v.table('transactions')
print(tx.shape)
print(tx[['sales_value','coupon_match_disc','coupon_disc','retail_disc','quantity']].describe().round(3).to_string())
print('neg frac:', {c: float((tx[c]<0).mean()) for c in ['sales_value','coupon_match_disc','coupon_disc','retail_disc']})
print('zero sales frac', float((tx.sales_value==0).mean()))
print('households in tx:', tx.household_key.nunique())
print('day range', tx.day.min(), tx.day.max())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    hh = view.households
    hh_idx = pd.Index(hh)
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(hh_idx)]
    day = tx['day'].values
    out = pd.DataFrame(index=hh_idx)

    def spend_in(lo, hi):
        m = (day > lo) & (day <= hi)
        if not m.any():
            return pd.Series(0.0, index=hh_idx)
        s = tx.loc[m].groupby('household_key')['sales_value'].sum()
        return s.reindex(hh_idx).fillna(0.0)

    def baskets_in(lo, hi):
        m = (day > lo) & (day <= hi)
        if not m.any():
            return pd.Series(0.0, index=hh_idx)
        s = tx.loc[m].groupby('household_key')['basket_id'].nunique()
        return s.reindex(hh_idx).fillna(0.0)

    # multi-lag 28d spend sequence (lag-2 and lag-3 windows)
    out['spend_28_prior2'] = spend_in(snapshot_day-84, snapshot_day-56)
    out['spend_28_prior3'] = spend_in(snapshot_day-112, snapshot_day-84)

    # trip-frequency trend: last-28d trips vs 84d run-rate
    b28 = baskets_in(snapshot_day-28, snapshot_day)
    b84 = baskets_in(snapshot_day-84, snapshot_day)
    out['trip_freq_ratio'] = b28 / (b84*(28.0/84.0) + 0.5)

    # typical (median) 28d-window spend over trailing 182d
    first_day = tx.groupby('household_key')['day'].min().reindex(hh_idx)
    wins = [spend_in(snapshot_day-28*k, snapshot_day-28*(k-1)) for k in range(1, 7)]
    W = pd.concat(wins, axis=1)
    starts = np.array([snapshot_day-28*k for k in range(1, 7)])
    mask = starts[None, :] >= first_day.values[:, None]
    W = W.where(pd.DataFrame(mask, index=hh_idx, columns=W.columns))
    out['med28_182'] = W.median(axis=1, skipna=True)

    # household-specific seasonal index: historical spend share in the upcoming window's week-of-year slots
    wk = tx['week_no'].values
    wy = (wk - 1) % 52
    twys = np.array(sorted({(((dd + 8)//7) - 1) % 52 for dd in range(snapshot_day+1, snapshot_day+29)}))
    m = np.isin(wy, twys)
    sm = tx.loc[m].groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    sa = tx.groupby('household_key')['sales_value'].sum().reindex(hh_idx).fillna(0.0)
    span = (snapshot_day - first_day)
    idx = sm / (sa * (len(twys)/52.0)).replace(0, np.nan)
    out['hh_seasonal_index'] = idx.where(span >= 180)

    if snapshot_day <= 431:
        print(snapshot_day, out.describe().loc[['mean','50%']].round(3).to_dict())
    return out

t = agent_api.build_features(fn)
print(t.shape, t.columns.tolist())
print(t.isna().mean().round(3).to_string())
p = agent_api.save_table(t, 'e009_ar_season.parquet')
print(p)
