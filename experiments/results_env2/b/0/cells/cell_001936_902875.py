
import agent_api, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')

tt = agent_api.train_targets()

def prep(df):
    feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    return X.astype(float)

def linear_proxy(df, lam=30):
    X = prep(df)
    m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
    y = m.future_spend_4w.values
    trm = (m.snapshot_day<=375).values; vam = m.snapshot_day.isin([403,431]).values
    Xtr, Xva = X[trm], X[vam]
    med = Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
    A = np.c_[np.ones(trm.sum()), ((Xtr-mu)/sd).values]; B = np.c_[np.ones(vam.sum()), ((Xva-mu)/sd).values]
    M = A.T@A + lam*np.eye(A.shape[1]); M[0,0]-=lam
    b = np.linalg.solve(M, A.T@y[trm])
    return np.abs(B@b - y[vam]).mean()

def gbm_proxy(df, depth=3, lr=0.1, n_trees=150, l2=1.0, min_leaf=20, seed=0):
    X = prep(df)
    m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
    y = m.future_spend_4w.values
    trm = (m.snapshot_day<=375).values; vam = m.snapshot_day.isin([403,431]).values
    Xtr, Xva = X[trm], X[vam]
    med = Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    rng = np.random.RandomState(seed)
    F = Xtr.shape[1]
    edges = [np.unique(np.quantile(Xtr[:,f], np.linspace(0,1,33)[1:-1])) for f in range(F)]
    def binmat(Xa):
        out = np.zeros((len(Xa), F), dtype=np.int8)
        for f in range(F):
            b = np.searchsorted(edges[f], Xa[:,f])
            b[np.isnan(Xa[:,f])] = 32
            out[:,f] = np.clip(b, 0, 32)
        return out
    Btr, Bva = binmat(Xtr), binmat(Xva)
    ytr = y[trm]; pred = np.zeros(len(ytr)); trees = []
    for t in range(n_trees):
        g = pred - ytr
        feats = rng.choice(F, int(F*0.7), replace=False)
        nodes = {'feat':[], 'thr':[], 'left':[], 'right':[], 'leaf':[]}
        def new_leaf(idx):
            nodes['leaf'].append(-g[idx].sum()/(len(idx)+l2)); nodes['feat'].append(-1)
            return len(nodes['leaf'])-1
        def grow(idx, d):
            Gf = np.zeros((F,33)); Hf = np.zeros((F,33))
            bf = Btr[idx]; w = g[idx]
            for f in feats:
                Gf[f] = np.bincount(bf[:,f], weights=w, minlength=33)
                Hf[f] = np.bincount(bf[:,f], minlength=33)
            Gl = np.cumsum(Gf,1); Hl = np.cumsum(Hf,1)
            Gr = Gf.sum(1)[:,None]-Gl; Hr = Hf.sum(1)[:,None]-Hl
            gain = Gl**2/(Hl+l2) + Gr**2/(Hr+l2) - Gf.sum(1)[:,None]**2/(Hf.sum(1)[:,None]+l2)
            bad = (Hl<min_leaf)|(Hr<min_leaf); gain[:, -1] = -1e18; gain[bad] = -1e18
            fbest = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[fbest] <= 1e-9 or d==depth or len(idx)<2*min_leaf:
                return new_leaf(idx)
            f, thr = fbest
            go_l = bf[:,f] <= thr
            nodes['feat'].append(f); nodes['thr'].append(thr); nodes['left'].append(-1); nodes['right'].append(-1)
            me = len(nodes['feat'])-1
            nodes['left'][me] = grow(idx[go_l], d+1); nodes['right'][me] = grow(idx[~go_l], d+1)
            return me
        root = grow(np.arange(len(ytr)), 0)
        featA = np.array(nodes['feat']); thrA = np.array(nodes['thr'])
        leftA = np.array(nodes['left']); rightA = np.array(nodes['right']); leafA = np.array(nodes['leaf'])
        trees.append((featA,thrA,leftA,rightA,leafA))
        # update train preds
        cur = np.zeros(len(ytr), dtype=int)
        while True:
            mask = featA[cur] >= 0
            if not mask.any(): break
            rows = np.where(mask)[0]; f = featA[cur[rows]]; b = Btr[rows, f]
            cur[rows] = np.where(b<=thrA[cur[rows]], leftA[cur[rows]], rightA[cur[rows]])
        upd = leafA[cur]; pred += lr*upd
    # val preds
    cur = np.zeros(len(Bva), dtype=int)
    out = np.zeros(len(Bva))
    for featA,thrA,leftA,rightA,leafA in trees:
        cur = np.zeros(len(Bva), dtype=int)
        while True:
            mask = featA[cur] >= 0
            if not mask.any(): break
            rows = np.where(mask)[0]; f = featA[cur[rows]]; b = Bva[rows, f]
            cur[rows] = np.where(b<=thrA[cur[rows]], leftA[cur[rows]], rightA[cur[rows]])
        out += lr*leafA[cur]
    yva = y[vam]
    return np.abs(out - yva).mean()

t0=time.time()
harness = {'e013':60.788,'e011':60.895,'e015':60.944,'e017':60.808,'e014':61.912,'e016':65.334}
for name in ['e013','e011','e015','e017','e014','e016']:
    path = {'e013':'e013_stationary.parquet','e011':'e011_pruned_basket.parquet','e015':'e015_norm_windows.parquet',
            'e017':'e017_reversion.parquet','e014':'e014_seasonal.parquet','e016':'e016_display.parquet'}[name]
    df = agent_api.load_saved(path)
    print('%s harness %.2f | linear %.2f' % (name, harness[name], linear_proxy(df)))
print('linear done %.0fs' % (time.time()-t0))
for name in ['e013','e014','e016']:
    path = {'e013':'e013_stationary.parquet','e014':'e014_seasonal.parquet','e016':'e016_display.parquet'}[name]
    df = agent_api.load_saved(path)
    print('%s harness %.2f | gbm %.2f' % (name, harness[name], gbm_proxy(df)))
print('all done %.0fs' % (time.time()-t0))
