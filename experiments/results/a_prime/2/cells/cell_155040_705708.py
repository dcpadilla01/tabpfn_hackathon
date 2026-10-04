
import agent_api, pandas as pd, numpy as np, datetime

sd = agent_api.snapshot_days()

def prep_matrix(t, drop_extra=()):
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and c not in drop_extra]
    def prep(df):
        X = df[cols].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        return X.astype(np.float64)
    return cols, prep

def fit_gbm(Xtr, ytr, Xva, n_trees=200, lr=0.1, max_depth=4, min_leaf=150, feat_frac=0.4, seed=0, nb=64):
    rng = np.random.RandomState(seed)
    n, p = Xtr.shape
    nv = Xva.shape[0]
    qs = np.linspace(0,1,nb+2)[1:-1]
    thr = np.zeros((p, nb-1), dtype=np.float64)
    for j in range(p):
        u = np.unique(np.quantile(Xtr[:,j], qs))
        thr[j,:len(u)] = u
        if len(u) < nb-1: thr[j,len(u):] = u[-1] if len(u) else 0.0
    Xb = np.empty((n,p), dtype=np.int32); Xvb = np.empty((nv,p), dtype=np.int32)
    for j in range(p):
        Xb[:,j] = np.searchsorted(thr[j], Xtr[:,j])
        Xvb[:,j] = np.searchsorted(thr[j], Xva[:,j])
    pred_va = np.full(nv, ytr.mean()); pred_tr = np.full(n, ytr.mean())
    for t_i in range(n_trees):
        res_tr = ytr - pred_tr
        tree = {}
        def build(idx, depth, nid):
            g = res_tr[idx]
            tree[nid] = {'leaf': g.mean()}
            if depth>=max_depth or len(idx)<2*min_leaf: return
            best = None
            feats = rng.choice(p, max(1,int(p*feat_frac)), replace=False)
            gc = g.sum(); nn=len(idx)
            base = (g**2).sum() - gc**2/nn
            for j in feats:
                hj = Xb[idx,j]
                cnt = np.bincount(hj, minlength=nb)
                s = np.bincount(hj, weights=g, minlength=nb)
                c = np.cumsum(cnt)[:-1]; ss = np.cumsum(s)[:-1]
                nl = c; nr = nn - c
                ok = (nl>=min_leaf)&(nr>=min_leaf)
                if not ok.any(): continue
                sl = ss; sr = gc - ss
                var = (sl**2/np.maximum(nl,1)) + (sr**2/np.maximum(nr,1))
                var[~ok] = -np.inf
                k = int(var.argmax())
                if var[k] > -np.inf and (best is None or var[k] > best[0]):
                    best = (var[k], j, k)
            if best is None or best[0] <= base - 1e-12: return
            _, j, k = best
            m = Xb[idx,j] <= k
            tree[nid] = {'feat': j, 'thr': thr[j][k], 'left': 2*nid+1, 'right': 2*nid+2}
            build(idx[m], depth+1, 2*nid+1); build(idx[~m], depth+1, 2*nid+2)
        build(np.arange(n), 0, 0)
        def predict(Xbin):
            n_ = len(Xbin)
            out = np.zeros(n_)
            nid = np.zeros(n_, dtype=np.int64)
            while True:
                cur = nid
                f = np.empty(n_, dtype=np.int64); f.fill(-1)
                thv = np.empty(n_)
                for i in range(n_):
                    nd = tree[int(cur[i])]
                    if 'feat' in nd:
                        f[i] = nd['feat']; thv[i] = nd['thr']
                leaf = f < 0
                out[leaf] += np.array([tree[int(cur[i])]['leaf'] for i in np.where(leaf)[0]])
                if leaf.all(): break
                act = np.where(~leaf)[0]
                bidx = np.array([np.searchsorted(thr[f[i]], thv[i]) for i in act])
                go_l = Xbin[act, f[act]] <= bidx
                nid[act] = np.where(go_l, [tree[int(n)]['left'] for n in nid[act]], [tree[int(n)]['right'] for n in nid[act]])
            return out
        dva = predict(Xvb); dtr = predict(Xb)
        pred_va += lr*dva; pred_tr += lr*dtr
    return pred_va, pred_tr

b = agent_api.baseline_features()
tt = agent_api.train_targets()
m = b.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
cols, prep = prep_matrix(b)
Xtr = prep(tr).values; Xva = prep(va).values
ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
med = np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
t0 = datetime.datetime.now()
pv,_ = fit_gbm(Xtr, ytr, Xva, n_trees=120)
print('E000 proxy GBM val MAE', round(np.abs(pv-yva).mean(),2), 'time', round((datetime.datetime.now()-t0).total_seconds(),1), '(harness: 92.4)')
