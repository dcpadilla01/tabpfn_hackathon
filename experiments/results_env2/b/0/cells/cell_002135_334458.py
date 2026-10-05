
import agent_api, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()

def prep(df):
    feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    X = X.astype(float).replace([np.inf,-np.inf], np.nan)
    return X

def gbm_proxy(df, depth=3, lr=0.05, n_trees=200, l2=5.0, min_leaf=30, seed=0):
    X = prep(df).values
    m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
    y = m.future_spend_4w.values
    trm = (m.snapshot_day<=375).values; vam = m.snapshot_day.isin([403,431]).values
    Xtr, Xva = X[trm], X[vam]
    med = np.nanmedian(Xtr, 0); Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    rng = np.random.RandomState(seed); F = Xtr.shape[1]
    edges = [np.unique(np.quantile(Xtr[:,f], np.linspace(0,1,33)[1:-1])) for f in range(F)]
    def binmat(Xa):
        out = np.zeros((len(Xa), F), dtype=np.int8)
        for f in range(F):
            b = np.searchsorted(edges[f], Xa[:,f]); b[np.isnan(Xa[:,f])] = 32
            out[:,f] = np.clip(b, 0, 32)
        return out
    Btr, Bva = binmat(Xtr), binmat(Xva)
    ytr = y[trm]; pred = np.zeros(len(ytr)); trees = []
    for t in range(n_trees):
        g = pred - ytr
        feats = rng.choice(F, max(2,int(F*0.7)), replace=False)
        featL=[]; thrL=[]; leftL=[]; rightL=[]; leafL=[]
        def new_leaf(idx):
            leafL.append(-g[idx].sum()/(len(idx)+l2)); featL.append(-1); thrL.append(0); leftL.append(-1); rightL.append(-1)
            return len(leafL)-1
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
            featL.append(int(f)); thrL.append(int(thr)); leftL.append(-1); rightL.append(-1)
            me = len(featL)-1
            leftL[me] = grow(idx[go_l], d+1); rightL[me] = grow(idx[~go_l], d+1)
            return me
        grow(np.arange(len(ytr)), 0)
        trees.append((np.array(featL),np.array(thrL),np.array(leftL),np.array(rightL),np.array(leafL)))
        cur = np.zeros(len(ytr), dtype=int)
        while True:
            mask = featL[cur] = featL[cur]; mask = np.array(featL)[cur] >= 0
            if not mask.any(): break
            rows = np.where(mask)[0]; f = np.array(featL)[cur[rows]]; b = Btr[rows, f]
            cur[rows] = np.where(b<=np.array(thrL)[cur[rows]], np.array(leftL)[cur[rows]], np.array(rightL)[cur[rows]])
        pred += lr*np.array(leafL)[cur]
    out = np.zeros(len(Bva))
    for featA,thrA,leftA,rightA,leafA in trees:
        cur = np.zeros(len(Bva), dtype=int)
        while True:
            mask = featA[cur] >= 0
            if not mask.any(): break
            rows = np.where(mask)[0]; f = featA[cur[rows]]; b = Bva[rows, f]
            cur[rows] = np.where(b<=thrA[cur[rows]], leftA[cur[rows]], rightA[cur[rows]])
        out += lr*leafA[cur]
    return np.abs(out - y[vam]).mean()

def linear_proxy(df, lam=30):
    X = prep(df).values
    m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
    y = m.future_spend_4w.values
    trm = (m.snapshot_day<=375).values; vam = m.snapshot_day.isin([403,431]).values
    Xtr, Xva = X[trm], X[vam]
    med = np.nanmedian(Xtr,0); Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.c_[np.ones(trm.sum()), (Xtr-mu)/sd]; B = np.c_[np.ones(vam.sum()), (Xva-mu)/sd]
    M = A.T@A + lam*np.eye(A.shape[1]); M[0,0]-=lam
    b = np.linalg.solve(M, A.T@y[trm])
    return np.abs(B@b - y[vam]).mean()

e13 = agent_api.load_saved('e013_stationary.parquet')
e17 = agent_api.load_saved('e017_reversion.parquet')
spike_cols = [c for c in e17.columns if c not in e13.columns]
spend_cols = [c for c in e13.columns if c.startswith(('spend','dec_','weekly_','avg_','z_rate84'))]

def mk(hinges=False, logs=False, spikes=False):
    X = e13.copy()
    if logs:
        for c in spend_cols: X[c] = np.log1p(X[c].clip(lower=0))
    if hinges:
        for col, thrs in {'dec_28':[10,25,50,100,200,400],'dec_56':[25,75,150,300],'spend_84':[25,75,200,500],
                          'z_rate84':[25,75,200],'weekly_mean_12':[25,75,200],'spend_182':[50,150,400]}.items():
            for t in thrs: X[col+'_h%d'%t] = np.maximum(X[col]-t, 0)
    if spikes:
        X = X.merge(e17[['household_key','snapshot_day']+spike_cols], on=['household_key','snapshot_day'], how='left')
    return X

t0=time.time()
harness = {'e013':60.788,'e011':60.895,'e015':60.944,'e017':60.808,'e014':61.912,'e016':65.334}
paths = {'e013':'e013_stationary.parquet','e011':'e011_pruned_basket.parquet','e015':'e015_norm_windows.parquet',
         'e017':'e017_reversion.parquet','e014':'e014_seasonal.parquet','e016':'e016_display.parquet'}
print('--- GBM proxy sanity ---')
for name in ['e013','e011','e017','e014','e016']:
    print('%s harness %.2f | gbm %.2f' % (name, harness[name], gbm_proxy(agent_api.load_saved(paths[name]))))
print('%.0fs' % (time.time()-t0))
print('--- candidates ---')
for label, kw in [('e13',{}),('e13+hinge',{'hinges':True}),('e13+log',{'logs':True}),
                  ('e13+spike',{'spikes':True}),('e13+hinge+spike',{'hinges':True,'spikes':True}),
                  ('e13+log+hinge+spike',{'logs':True,'hinges':True,'spikes':True})]:
    X = mk(**kw)
    print('%-20s gbm %.2f | linear %.2f' % (label, gbm_proxy(X), linear_proxy(X)))
print('%.0fs' % (time.time()-t0))
