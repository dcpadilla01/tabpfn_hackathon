
import agent_api, pandas as pd, numpy as np, datetime

sd = agent_api.snapshot_days()

def prep_matrix(t, drop_extra=('index',)):
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and c not in drop_extra]
    def prep(df):
        X = df[cols].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        return X.astype(np.float64)
    return cols, prep

def fit_gbm(Xtr, ytr, Xva, n_trees=250, lr=0.08, max_depth=4, min_leaf=200, feat_frac=0.35, seed=0, nb=48):
    rng = np.random.RandomState(seed)
    n, p = Xtr.shape; nv = Xva.shape[0]
    qs = np.linspace(0,1,nb+2)[1:-1]
    thr = np.zeros((p, nb-1))
    for j in range(p):
        u = np.unique(np.quantile(Xtr[:,j], qs))
        thr[j,:len(u)] = u
        if len(u): thr[j,len(u):] = u[-1]
    Xb = np.empty((n,p), dtype=np.int32); Xvb = np.empty((nv,p), dtype=np.int32)
    for j in range(p):
        Xb[:,j] = np.searchsorted(thr[j], Xtr[:,j])
        Xvb[:,j] = np.searchsorted(thr[j], Xva[:,j])
    pred_va = np.full(nv, ytr.mean()); pred_tr = np.full(n, ytr.mean())
    for t_i in range(n_trees):
        res = ytr - pred_tr
        nodes = {}; queue = [(np.arange(n), 0, 0)]
        order = []
        while queue:
            idx, depth, nid = queue.pop(0)
            g = res[idx]
            if depth >= max_depth or len(idx) < 2*min_leaf:
                nodes[nid] = ('leaf', g.mean()); continue
            best = None
            gc = g.sum(); nn = len(idx)
            base = (g**2).sum() - gc**2/nn
            for j in rng.choice(p, max(1,int(p*feat_frac)), replace=False):
                hj = Xb[idx,j]
                cnt = np.bincount(hj, minlength=nb)
                s = np.bincount(hj, weights=g, minlength=nb)
                c = np.cumsum(cnt)[:-1]; ss = np.cumsum(s)[:-1]
                nl = c; nr = nn - c
                ok = (nl>=min_leaf)&(nr>=min_leaf)
                if not ok.any(): continue
                var = (ss**2/np.maximum(nl,1)) + ((gc-ss)**2/np.maximum(nr,1))
                var[~ok] = -np.inf
                k = int(var.argmax())
                if var[k] > -np.inf and (best is None or var[k] > best[0]):
                    best = (var[k], j, k)
            if best is None or best[0] <= base - 1e-12:
                nodes[nid] = ('leaf', g.mean()); continue
            _, j, k = best
            m = Xb[idx,j] <= k
            nodes[nid] = ('split', j, k, 2*nid+1, 2*nid+2)
            order.append(nid)
            queue.append((idx[m], depth+1, 2*nid+1)); queue.append((idx[~m], depth+1, 2*nid+2))
        def predict(Xbin):
            nid = np.zeros(len(Xbin), dtype=np.int64)
            for nid_ in order:
                _, j, k, l, r = nodes[nid_]
                msk = nid == nid_
                go_l = Xbin[msk, j] <= k
                nid[msk] = np.where(go_l, l, r)
            return np.array([nodes[i][1] for i in nid])
        pred_va += lr*predict(Xvb); pred_tr += lr*predict(Xb)
    return pred_va

def run(name, t):
    tt = agent_api.train_targets()
    m = t.merge(tt, on=['household_key','snapshot_day'])
    tr_fit = m[m.snapshot_day.isin(sd['train'][:-1])]   # 95..403
    tr_ev  = m[m.snapshot_day==431]
    cols, prep = prep_matrix(t)
    Xf = prep(tr_fit); keep = Xf.columns[Xf.notna().any()].tolist()
    Xf = Xf[keep].values; Xe = prep(tr_ev)[keep].values
    yf = tr_fit['future_spend_4w'].values; ye = tr_ev['future_spend_4w'].values
    med = np.nanmedian(Xf,0); med = np.where(np.isnan(med),0,med)
    Xf = np.where(np.isnan(Xf), med, Xf); Xe = np.where(np.isnan(Xe), med, Xe)
    t0 = datetime.datetime.now()
    pv = fit_gbm(Xf, yf, Xe)
    print(f'{name}: inner431 MAE {np.abs(pv-ye).mean():.2f}  nfeat {len(keep)}  ({(datetime.datetime.now()-t0).total_seconds():.0f}s)')

run('E000', agent_api.baseline_features())
run('E001', agent_api.load_saved('recency_agg.parquet'))
run('E005', agent_api.load_saved('basket_tenure.parquet'))
run('E009', agent_api.load_saved('e009_ewma_longlags.parquet'))
