import agent_api, numpy as np, pandas as pd
t = agent_api.load_saved('e008_level_shape.parquet')
print('E008 shape', t.shape)
feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print('n_feat', len(feat))
print('dtypes', t[feat].dtypes.value_counts().to_dict())
print('cols:', feat)
tt = agent_api.train_targets()
print('train_targets', tt.shape)
print({k: round(v,2) for k,v in tt.future_spend_4w.describe().to_dict().items()})
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape, 'snapdays', sorted(m.snapshot_day.unique()))
y = m.future_spend_4w.values.astype(float)
print('zero frac', round((y==0).mean(),3))
Xdf = m[feat].copy()
for c in feat:
    dt = str(Xdf[c].dtype)
    if dt in ('object','category','bool'):
        Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan)
X = Xdf.values.astype(float)
print('X', X.shape, 'nanfrac', round(np.isnan(X).mean(),4))

def prep(Xtr, Xev):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0.0, med)
    A = np.where(np.isnan(Xtr), med, Xtr); B = np.where(np.isnan(Xev), med, Xev)
    mu = A.mean(0); sd = A.std(0); sd[sd<1e-12]=1
    return (A-mu)/sd, (B-mu)/sd

def ridge_fit(Z, yy, a):
    p = Z.shape[1]
    return np.linalg.solve(Z.T@Z + a*np.eye(p), Z.T@yy)

folds = [375, 403, 431]
for logt in [False, True]:
    for a in [3,10,30,100,300]:
        maes=[]
        for e in folds:
            tr = m.snapshot_day.values < e; ev = m.snapshot_day.values == e
            Ztr, Zev = prep(X[tr], X[ev])
            yy = np.log1p(y) if logt else y
            w = ridge_fit(Ztr, yy[tr], a)
            pr = Zev@w
            if logt: pr = np.expm1(np.clip(pr, 0, 12))
            maes.append(np.abs(pr - y[ev]).mean())
        print('logt' if logt else 'raw ', 'a', a, 'MAE', np.round(maes,2), 'avg', round(np.mean(maes),3))
for e in folds:
    tr = m.snapshot_day.values < e; ev = m.snapshot_day.values == e
    print('meanbase', e, round(np.abs(y[tr].mean()-y[ev]).mean(),3))
Xf = np.where(np.isnan(X), np.nanmedian(X,0), X)
cs = np.array([abs(np.corrcoef(Xf[:,j], y)[0,1]) if np.std(Xf[:,j])>0 else 0 for j in range(X.shape[1])])
order = np.argsort(-cs)
print('top25 |corr|:')
for j in order[:25]: print('  ', feat[j], round(cs[j],3))
for K in [20,40,60,80,120,163]:
    sub = order[:K]
    maes=[]
    for e in folds:
        tr = m.snapshot_day.values < e; ev = m.snapshot_day.values == e
        Ztr, Zev = prep(X[tr][:,sub], X[ev][:,sub])
        w = ridge_fit(Ztr, y[tr], 30)
        maes.append(np.abs(Zev@w - y[ev]).mean())
    print('topK', K, 'ridge a30 raw MAE', round(np.mean(maes),3))


# ---- cell ----
import agent_api, numpy as np, pandas as pd, time

t = agent_api.load_saved('e008_level_shape.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'])
feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xdf = m[feat].copy()
for c in feat:
    if str(Xdf[c].dtype) in ('object','category','bool'):
        Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
    else:
        Xdf[c] = Xdf[c].astype(float)
X = Xdf.values; y = m.future_spend_4w.values.astype(float); sd = m.snapshot_day.values.astype(int)

def binize(Xref, Xnew, nbins=32):
    p = Xref.shape[1]
    edges = []; Bref = np.zeros(Xref.shape, np.int16); Bnew = np.zeros(Xnew.shape, np.int16)
    for j in range(p):
        col = Xref[:,j]; med = np.nanmedian(col)
        if np.isnan(med): med = 0.0
        colf = np.where(np.isnan(col), med, col)
        qs = np.unique(np.quantile(colf, np.linspace(0,1,nbins+1)[1:-1]))
        edges.append(qs)
        Bref[:,j] = np.searchsorted(qs, colf, 'right')
        coln = Xnew[:,j]; coln = np.where(np.isnan(coln), med, coln)
        Bnew[:,j] = np.searchsorted(qs, coln, 'right')
    return Bref, Bnew, edges

def gbm_fit(Btr, ytr, edges, depth=4, T=150, lr=0.1, lam=1.0, minleaf=20, seed=0, sub=0.85):
    rng = np.random.RandomState(seed)
    n, p = Btr.shape
    nb = np.array([len(e) for e in edges]); off = np.concatenate([[0], np.cumsum(nb+1)[:-1]])
    total = int(off[-1] + nb[-1] + 1)
    F = np.zeros(n); trees = []
    for t in range(T):
        resid = ytr - F
        rows0 = rng.choice(n, int(n*sub), replace=False)
        leafval = {}; splits = {}
        queue = [(rows0, 0, 0)]
        while queue:
            rows, d, nid = queue.pop()
            g = resid[rows]; G = g.sum(); H = len(rows)
            if d >= depth or H < 2*minleaf:
                leafval[nid] = lr * (-G/(H+lam)); F[rows] += leafval[nid]; continue
            subM = Btr[rows]
            flat = (subM + off[None,:]).ravel()
            w = np.repeat(g, p)
            hist = np.bincount(flat, weights=w, minlength=total)
            cnt = np.bincount(flat, minlength=total)
            bg, bj, bb = 0.0, -1, -1
            for j in range(p):
                h = hist[off[j]:off[j]+nb[j]+1]; c = cnt[off[j]:off[j]+nb[j]+1]
                if len(h) < 2: continue
                GL = np.cumsum(h)[:-1]; HL = np.cumsum(c)[:-1]
                GR = G - GL; HR = H - HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - G*G/(H+lam)
                gain[(HL<minleaf)|(HR<minleaf)] = -1
                b = int(np.argmax(gain))
                if gain[b] > bg: bg, bj, bb = gain[b], j, b
            if bj < 0:
                leafval[nid] = lr * (-G/(H+lam)); F[rows] += leafval[nid]; continue
            mask = subM[:, bj] <= bb
            lid, rid = nid*2+1, nid*2+2
            splits[nid] = (bj, bb, lid, rid)
            queue.append((rows[mask], d+1, lid)); queue.append((rows[~mask], d+1, rid))
        trees.append((splits, leafval))
    return trees

def gbm_predict(Bev, trees):
    out = np.zeros(Bev.shape[0])
    for splits, leafval in trees:
        active = {0: np.arange(Bev.shape[0])}
        while active:
            nid, rows = active.popitem()
            if nid in leafval: out[rows] += leafval[nid]; continue
            j, b, lid, rid = splits[nid]
            msk = Bev[rows, j] <= b
            if msk.any(): active[lid] = np.concatenate([active.get(lid, np.array([],int)), rows[msk]])
            if (~msk).any(): active[rid] = np.concatenate([active.get(rid, np.array([],int)), rows[~msk]])
    return out

def run_cv(cols_idx, tag, T=150):
    maes = []
    for tr_max, evs in [(347,[375]), (375,[403,431])]:
        tr = sd <= tr_max; ev = np.isin(sd, evs)
        Btr, Bev, edges = binize(X[tr][:,cols_idx], X[ev][:,cols_idx])
        t0 = time.time()
        trees = gbm_fit(Btr, y[tr], edges, T=T)
        pr = gbm_predict(Bev, trees)
        mae = np.abs(pr - y[ev]).mean()
        maes.append(mae)
        print(f'{tag} fold(tr<={tr_max}) MAE {mae:.2f} ({time.time()-t0:.0f}s)')
    print(f'{tag} AVG {np.mean(maes):.3f}')
    return np.mean(maes)

trmask = sd <= 347
Xf = np.where(np.isnan(X), 0, X)
cs = np.array([abs(np.corrcoef(Xf[trmask,j], y[trmask])[0,1]) if np.std(Xf[trmask,j])>0 else 0 for j in range(X.shape[1])])
order = np.argsort(-cs)
run_cv(order[:80], 'top80')
run_cv(order[:40], 'top40')


# ---- cell ----
import agent_api, numpy as np, pandas as pd, time

t = agent_api.load_saved('e008_level_shape.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'])
feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xdf = m[feat].copy()
for c in feat:
    if str(Xdf[c].dtype) in ('object','category','bool'):
        Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
    else:
        Xdf[c] = Xdf[c].astype(float)
X = Xdf.values; y = m.future_spend_4w.values.astype(float); sd = m.snapshot_day.values.astype(int)

def binize(Xref, Xnew, nbins=32):
    p = Xref.shape[1]
    edges = []; Bref = np.zeros(Xref.shape, np.int16); Bnew = np.zeros(Xnew.shape, np.int16)
    for j in range(p):
        col = Xref[:,j]; med = np.nanmedian(col)
        if np.isnan(med): med = 0.0
        colf = np.where(np.isnan(col), med, col)
        qs = np.unique(np.quantile(colf, np.linspace(0,1,nbins+1)[1:-1]))
        edges.append(qs)
        Bref[:,j] = np.searchsorted(qs, colf, 'right')
        coln = Xnew[:,j]; coln = np.where(np.isnan(coln), med, coln)
        Bnew[:,j] = np.searchsorted(qs, coln, 'right')
    return Bref, Bnew, edges

tr = sd <= 347; ev = sd == 375
Btr, Bev, edges = binize(X[tr], X[ev])
nb = np.array([len(e) for e in edges]); off = np.concatenate([[0], np.cumsum(nb+1)[:-1]])
total = int(off[-1] + nb[-1] + 1)
print('nb sum', nb.sum(), 'total', total, 'max flat', (Btr.astype(int)+off[None,:]).max())

# tiny fit debug: T=3, depth=3, no subsample
def fit_debug(Btr, ytr, depth=3, T=3, lr=0.1, lam=1.0, minleaf=20):
    n, p = Btr.shape
    F = np.zeros(n); trees = []
    for it in range(T):
        resid = ytr - F
        print('iter', it, 'resid absmax', np.abs(resid).max())
        leafval = {}; splits = {}; queue = [(np.arange(n), 0, 0)]
        while queue:
            rows, d, nid = queue.pop()
            g = resid[rows]; G = g.sum(); H = len(rows)
            if d >= depth or H < 2*minleaf:
                leafval[nid] = lr * (-G/(H+lam)); F[rows] += leafval[nid]; continue
            subM = Btr[rows]
            flat = (subM.astype(np.int64) + off[None,:]).ravel()
            w = np.repeat(g, p)
            hist = np.bincount(flat, weights=w, minlength=total)
            cnt = np.bincount(flat, minlength=total)
            bg, bj, bb = 0.0, -1, -1
            for j in range(p):
                h = hist[off[j]:off[j]+nb[j]+1]; c = cnt[off[j]:off[j]+nb[j]+1]
                if len(h) < 2: continue
                GL = np.cumsum(h)[:-1]; HL = np.cumsum(c)[:-1]
                GR = G - GL; HR = H - HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - G*G/(H+lam)
                gain[(HL<minleaf)|(HR<minleaf)] = -1
                b = int(np.argmax(gain))
                if gain[b] > bg: bg, bj, bb = gain[b], j, b
            if bj < 0:
                leafval[nid] = lr * (-G/(H+lam)); F[rows] += leafval[nid]; continue
            mask = subM[:, bj] <= bb
            lid, rid = nid*2+1, nid*2+2
            splits[nid] = (bj, bb, lid, rid)
            queue.append((rows[mask], d+1, lid)); queue.append((rows[~mask], d+1, rid))
        trees.append((splits, leafval))
        print('  F absmax', np.abs(F).max(), 'y absmax', np.abs(ytr).max())
    return trees

trees = fit_debug(Btr, y[tr])
# predict check
def pred_tree(Bev, splits, leafval):
    out = np.zeros(Bev.shape[0])
    active = {0: np.arange(Bev.shape[0])}
    guard = 0
    while active:
        nid, rows = active.popitem()
        guard += 1
        if guard > 100000: print('GUARD HIT'); break
        if nid in leafval: out[rows] += leafval[nid]; continue
        j, b, lid, rid = splits[nid]
        msk = Bev[rows, j] <= b
        if msk.any(): active[lid] = np.concatenate([active.get(lid, np.array([],int)), rows[msk]])
        if (~msk).any(): active[rid] = np.concatenate([active.get(rid, np.array([],int)), rows[~msk]])
    return out
pr = np.zeros(Bev.shape[0])
for splits, leafval in trees:
    pr += pred_tree(Bev, splits, leafval)
print('pred stats', np.nanmin(pr), np.nanmax(pr), np.nanmean(pr))
print('MAE', np.abs(pr - y[ev]).mean())


# ---- cell ----
import agent_api, numpy as np, pandas as pd, time

t = agent_api.load_saved('e008_level_shape.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'])
feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xdf = m[feat].copy()
for c in feat:
    if str(Xdf[c].dtype) in ('object','category','bool'):
        Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
    else:
        Xdf[c] = Xdf[c].astype(float)
X = Xdf.values; y = m.future_spend_4w.values.astype(float); sd = m.snapshot_day.values.astype(int)

def binize(Xref, Xnew, nbins=32):
    p = Xref.shape[1]
    edges = []; Bref = np.zeros(Xref.shape, np.int16); Bnew = np.zeros(Xnew.shape, np.int16)
    for j in range(p):
        col = Xref[:,j]; med = np.nanmedian(col)
        if np.isnan(med): med = 0.0
        colf = np.where(np.isnan(col), med, col)
        qs = np.unique(np.quantile(colf, np.linspace(0,1,nbins+1)[1:-1]))
        edges.append(qs)
        Bref[:,j] = np.searchsorted(qs, colf, 'right')
        coln = Xnew[:,j]; coln = np.where(np.isnan(coln), med, coln)
        Bnew[:,j] = np.searchsorted(qs, coln, 'right')
    return Bref, Bnew, edges

def gbm_fit(Btr, ytr, edges, depth=4, T=150, lr=0.1, lam=1.0, minleaf=20, seed=0, sub=0.85):
    rng = np.random.RandomState(seed)
    n, p = Btr.shape
    nb = np.array([len(e) for e in edges]); off = np.concatenate([[0], np.cumsum(nb+1)[:-1]])
    total = int(off[-1] + nb[-1] + 1)
    F = np.zeros(n); trees = []
    for it in range(T):
        resid = ytr - F
        rows0 = rng.choice(n, int(n*sub), replace=False)
        leafval = {}; splits = {}
        queue = [(rows0, 0, 0)]
        while queue:
            rows, d, nid = queue.pop()
            g = resid[rows]; G = g.sum(); H = len(rows)
            if d >= depth or H < 2*minleaf:
                leafval[nid] = lr * (G/(H+lam)); F[rows] += leafval[nid]; continue
            subM = Btr[rows]
            flat = (subM.astype(np.int64) + off[None,:]).ravel()
            w = np.repeat(g, p)
            hist = np.bincount(flat, weights=w, minlength=total)
            cnt = np.bincount(flat, minlength=total)
            bg, bj, bb = 0.0, -1, -1
            for j in range(p):
                h = hist[off[j]:off[j]+nb[j]+1]; c = cnt[off[j]:off[j]+nb[j]+1]
                if len(h) < 2: continue
                GL = np.cumsum(h)[:-1]; HL = np.cumsum(c)[:-1]
                GR = G - GL; HR = H - HL
                gain = GL*GL/(HL+lam) + GR*GR/(HR+lam) - G*G/(H+lam)
                gain[(HL<minleaf)|(HR<minleaf)] = -1
                b = int(np.argmax(gain))
                if gain[b] > bg: bg, bj, bb = gain[b], j, b
            if bj < 0:
                leafval[nid] = lr * (G/(H+lam)); F[rows] += leafval[nid]; continue
            mask = subM[:, bj] <= bb
            lid, rid = nid*2+1, nid*2+2
            splits[nid] = (bj, bb, lid, rid)
            queue.append((rows[mask], d+1, lid)); queue.append((rows[~mask], d+1, rid))
        trees.append((splits, leafval))
    return trees

def gbm_predict(Bev, trees):
    out = np.zeros(Bev.shape[0])
    for splits, leafval in trees:
        active = {0: np.arange(Bev.shape[0])}
        while active:
            nid, rows = active.popitem()
            if nid in leafval: out[rows] += leafval[nid]; continue
            j, b, lid, rid = splits[nid]
            msk = Bev[rows, j] <= b
            if msk.any(): active[lid] = np.concatenate([active.get(lid, np.array([],int)), rows[msk]])
            if (~msk).any(): active[rid] = np.concatenate([active.get(rid, np.array([],int)), rows[~msk]])
    return out

def run_cv(cols_idx, tag, depth=4, T=150, lr=0.1, lam=1.0, minleaf=20):
    maes = []
    for tr_max, evs in [(347,[375]), (375,[403,431])]:
        tr = sd <= tr_max; ev = np.isin(sd, evs)
        Btr, Bev, edges = binize(X[tr][:,cols_idx], X[ev][:,cols_idx])
        t0 = time.time()
        trees = gbm_fit(Btr, y[tr], edges, depth=depth, T=T, lr=lr, lam=lam, minleaf=minleaf)
        pr = gbm_predict(Bev, trees)
        mae = np.abs(pr - y[ev]).mean()
        maes.append(mae)
        print(f'{tag} fold(tr<={tr_max}) MAE {mae:.2f} ({time.time()-t0:.0f}s)')
    print(f'{tag} AVG {np.mean(maes):.3f}')
    return np.mean(maes)

trmask = sd <= 347
Xf = np.where(np.isnan(X), 0, X)
cs = np.array([abs(np.corrcoef(Xf[trmask,j], y[trmask])[0,1]) if np.std(Xf[trmask,j])>0 else 0 for j in range(X.shape[1])])
order = np.argsort(-cs)
run_cv(order[:80], 'top80')
run_cv(order[:40], 'top40')


# ---- cell ----
import agent_api, numpy as np, pandas as pd

tt = agent_api.train_targets()

def prep(path):
    t = agent_api.load_saved(path)
    feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    m = t.merge(tt, on=['household_key','snapshot_day'], how='left')
    isval = m.future_spend_4w.isna().values
    Xdf = m[feat].copy()
    for c in feat:
        if str(Xdf[c].dtype) in ('object','category','bool'):
            Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            Xdf[c] = Xdf[c].astype(float)
    return feat, Xdf.values, isval

print('=== E009 TE features sanity ===')
f9, X9, v9 = prep('e009_target_enc.parquet')
te_cols = [c for c in f9 if c.startswith('te_') or 'te' in c.lower()][:12]
print('te-like cols:', te_cols)
idx = [f9.index(c) for c in te_cols]
for c,i in zip(te_cols, idx):
    a, b = X9[~v9, i], X9[v9, i]
    print(f'{c:28s} nanT={np.isnan(a).mean():.3f} nanV={np.isnan(b).mean():.3f} meanT={np.nanmean(a):8.2f} meanV={np.nanmean(b):8.2f} stdT={np.nanstd(a):7.2f} stdV={np.nanstd(b):7.2f}')

print()
print('=== E008 train vs validation-row distribution shifts (top 15) ===')
f8, X8, v8 = prep('e008_level_shape.parquet')
shifts = []
for j,c in enumerate(f8):
    a, b = X8[~v8, j], X8[v8, j]
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a)<10 or len(b)<10: continue
    sd = np.sqrt(np.nanvar(X8[:,j]) + 1e-9)
    shifts.append((abs(a.mean()-b.mean())/sd, c, a.mean(), b.mean()))
shifts.sort(reverse=True)
for s,c,ma,mb in shifts[:15]:
    print(f'{c:24s} shift={s:.2f}  train={ma:9.3f} val={mb:9.3f}')
print()
print('=== E004 marketing features: nan/means train vs val (first 12) ===')
f4, X4, v4 = prep('e004_marketing.parquet')
mk = [c for c in f4 if c not in f8][:12]
for c in mk:
    i = f4.index(c)
    a, b = X4[~v4, i], X4[v4, i]
    print(f'{c:24s} nanT={np.isnan(a).mean():.3f} nanV={np.isnan(b).mean():.3f} meanT={np.nanmean(a):7.3f} meanV={np.nanmean(b):7.3f}')


# ---- cell ----
import agent_api, numpy as np, pandas as pd, time
t0 = time.time()

def build_group(view, snapshot_day):
    D = int(snapshot_day)
    hh = view.households
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)

    # ---------- dept mapping ----------
    prod = view.table('products')
    dep = prod[['product_id','department']].drop_duplicates('product_id')
    tx = tx.merge(dep, on='product_id', how='left')
    tx['department'] = tx['department'].fillna('UNK')

    m28 = tx.day >= D-27
    m84 = tx.day >= D-83
    m14 = tx.day >= D-13

    # ---------- A: dept momentum (sh28 - sh84) + sh28 top depts ----------
    s28 = tx[m28].groupby('department').sales_value.sum()
    s84 = tx[m84].groupby('department').sales_value.sum()
    topd = list(s84.sort_values(ascending=False).head(10).index)
    t28h = tx[m28].groupby(['household_key','department']).sales_value.sum()
    t84h = tx[m84].groupby(['household_key','department']).sales_value.sum()
    sh28 = t28h.unstack('department').reindex(columns=topd)
    sh84 = t84h.unstack('department').reindex(columns=topd)
    sh28 = sh28.div(sh28.sum(1)+1e-9, axis=0).reindex(hh)
    sh84 = sh84.div(sh84.sum(1)+1e-9, axis=0).reindex(hh)
    mom = (sh28.fillna(0) - sh84.fillna(0))
    for d in topd:
        out['dm_mom_'+d] = mom[d].values
    for d in topd[:5]:
        out['dm_sh28_'+d] = sh28[d].fillna(0).values

    # ---------- C: habit / repeat ----------
    t28 = tx[m28]
    tprev = tx[(tx.day >= D-111) & (tx.day < D-27)]
    pairs = tprev[['household_key','product_id']].drop_duplicates()
    rep = t28.merge(pairs, on=['household_key','product_id'], how='left', indicator=True)
    rep28 = rep[rep['_merge']=='both'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    sp28h = t28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_repeat_share_28'] = (rep28/(sp28h+1e-9)).values
    n28 = t28.groupby('household_key').product_id.nunique().reindex(hh)
    nrep = rep[rep['_merge']=='both'].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['hb_rep_prod_frac_28'] = (nrep/(n28+1e-9)).values
    # dept entropy 84d
    d84 = t84h.unstack('department').reindex(hh).fillna(0)
    p = d84.div(d84.sum(1)+1e-9, axis=0).clip(lower=1e-9)
    out['hb_dept_entropy_84'] = (-(p*np.log(p)).sum(1)).values
    # top1 product share 84d
    gp84 = tx[m84].groupby(['household_key','product_id']).sales_value.sum()
    mx = gp84.groupby('household_key').max().reindex(hh)
    tot = gp84.groupby('household_key').sum().reindex(hh)
    out['hb_top1prod_share_84'] = (mx/(tot+1e-9)).values
    # exploratory: spend in 28d on products not seen in 364d
    told = tx[tx.day >= D-363]
    oldp = told[['household_key','product_id']].drop_duplicates()
    newr = t28.merge(oldp, on=['household_key','product_id'], how='left', indicator=True)
    newsp = newr[newr['_merge']=='left_only'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_new_prod_share_28'] = (newsp/(sp28h+1e-9)).values

    # ---------- D: stock-up basket shape ----------
    b = tx.groupby(['household_key','basket_id']).agg(bspend=('sales_value','sum'), bday=('day','first'), bt=('trans_time','mean'))
    b84 = b[b.bday >= D-83]
    g = b84.groupby('household_key')
    bmax = g.bspend.max().reindex(hh); bmed = g.bspend.median().reindex(hh)
    bsum = g.bspend.sum().reindex(hh)
    out['su_max_over_med_84'] = (bmax/(bmed+1e-9)).values
    bidx = g.bspend.idxmax()
    bdaymap = b84.bday
    out['su_days_since_maxb_84'] = (D - bidx.map(bdaymap)).reindex(hh).fillna(999).values
    out['su_top1basket_share_84'] = (bmax/(bsum+1e-9)).values
    out['su_bigbasket_cnt_84'] = b84.assign(bb=b84.bspend > 2*(bmed+1e-9)).groupby('household_key').bb.sum().reindex(hh).fillna(0).values
    out['su_stockup_share_84'] = b84.assign(bb=b84.bspend > 2*(bmed+1e-9)).assign(sv=lambda x: x.bspend*x.bb).groupby('household_key').sv.sum().reindex(hh).fillna(0).values/(bsum+1e-9)
    q28 = t28.groupby('household_key').quantity.sum().reindex(hh)
    tr28 = t28.groupby('household_key').basket_id.nunique().reindex(hh)
    q84 = tx[m84].groupby('household_key').quantity.sum().reindex(hh)
    tr84 = tx[m84].groupby('household_key').basket_id.nunique().reindex(hh)
    out['su_qtyrate_28_84'] = ((q28/(tr28+1e-9))/((q84/(tr84+1e-9))+1e-9)).replace([np.inf,-np.inf],np.nan).values

    # ---------- J: time of day ----------
    tt = b84.bt
    w = b84.bspend
    am = b84[tt < 1200].groupby('household_key').bspend.sum().reindex(hh).fillna(0)
    ev = b84[tt >= 1800].groupby('household_key').bspend.sum().reindex(hh).fillna(0)
    out['td_am_share_84'] = (am/(bsum+1e-9)).values
    out['td_eve_share_84'] = (ev/(bsum+1e-9)).values
    out['td_avg_time_84'] = g.bt.mean().reindex(hh).values

    # ---------- B: global level / trend / seasonal index (scalars) ----------
    tx2 = tx.assign(wk=(tx.day+8)//7)
    gw = tx2.groupby('wk').sales_value.sum()
    ga = tx2.groupby('wk').household_key.nunique()
    gwk = (gw/(ga+1e-9))
    cw = (D+8)//7
    def wmean(a, b_):
        sel = gwk.loc[a:b_]
        return float(sel.mean()) if len(sel) else np.nan
    g_level = wmean(max(1,cw-8), cw-1)
    g_prev = wmean(max(1,cw-16), cw-9)
    g_trend = g_level/(g_prev+1e-9) if g_prev and not np.isnan(g_prev) else np.nan
    w0 = cw+1
    seas = np.nan; seas_rel = np.nan
    if w0-52 >= 1:
        seas = wmean(w0-52, w0-49)
        if not np.isnan(seas) and not np.isnan(g_level):
            seas_rel = seas/(g_level+1e-9)
    n = len(hh)
    out['g_level'] = np.full(n, g_level if not np.isnan(g_level) else 0.0)
    out['g_trend'] = np.full(n, g_trend if not np.isnan(g_trend) else 1.0)
    out['g_season_rel'] = np.full(n, seas_rel)
    out['g_season_abs'] = np.full(n, seas)
    return out

df = agent_api.build_features(build_group)
print('built', df.shape, 'cols', list(df.columns)[:6], '... total new feat', df.shape[1]-2, 'time', round(time.time()-t0))

e8 = agent_api.load_saved('e008_level_shape.parquet')
groups = {
 'candA_dm': ['dm_'], 'candB_g': ['g_'], 'candC_hb': ['hb_'], 'candD_su': ['su_'], 'candE_td': ['td_'],
}
for name, prefs in groups.items():
    cols = ['household_key','snapshot_day'] + [c for c in df.columns if any(c.startswith(p) for p in prefs)]
    sub = df[cols]
    mg = e8.merge(sub, on=['household_key','snapshot_day'], how='left')
    assert len(mg) == len(e8), (name, len(mg), len(e8))
    p = agent_api.save_table(mg, name)
    print(name, mg.shape, 'nan%', round(mg[cols[2:]].isna().mean().mean()*100,1))
print('done', round(time.time()-t0), 's')


# ---- cell ----
import agent_api, numpy as np, pandas as pd, time
t0 = time.time()

def build_group(view, snapshot_day):
    D = int(snapshot_day)
    hh = view.households
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)

    prod = view.table('products')
    dep = prod[['product_id','department']].drop_duplicates('product_id')
    dep = dep.assign(department=dep.department.astype(str))
    tx = tx.merge(dep, on='product_id', how='left')
    tx['department'] = tx['department'].astype(object).where(tx['department'].notna(), 'UNK')

    m28 = tx.day >= D-27
    m84 = tx.day >= D-83

    # A: dept momentum
    s84 = tx[m84].groupby('department').sales_value.sum()
    topd = list(s84.sort_values(ascending=False).head(10).index)
    t28h = tx[m28].groupby(['household_key','department']).sales_value.sum()
    t84h = tx[m84].groupby(['household_key','department']).sales_value.sum()
    sh28 = t28h.unstack('department').reindex(columns=topd)
    sh84 = t84h.unstack('department').reindex(columns=topd)
    sh28 = sh28.div(sh28.sum(1)+1e-9, axis=0).reindex(hh)
    sh84 = sh84.div(sh84.sum(1)+1e-9, axis=0).reindex(hh)
    mom = (sh28.fillna(0) - sh84.fillna(0))
    for d in topd:
        out['dm_mom_'+d] = mom[d].values
    for d in topd[:5]:
        out['dm_sh28_'+d] = sh28[d].fillna(0).values

    # C: habit / repeat
    t28 = tx[m28]
    tprev = tx[(tx.day >= D-111) & (tx.day < D-27)]
    pairs = tprev[['household_key','product_id']].drop_duplicates()
    rep = t28.merge(pairs, on=['household_key','product_id'], how='left', indicator=True)
    rep28 = rep[rep['_merge']=='both'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    sp28h = t28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_repeat_share_28'] = (rep28/(sp28h+1e-9)).values
    n28 = t28.groupby('household_key').product_id.nunique().reindex(hh)
    nrep = rep[rep['_merge']=='both'].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['hb_rep_prod_frac_28'] = (nrep/(n28+1e-9)).values
    d84 = t84h.unstack('department').reindex(hh).fillna(0)
    p_ = d84.div(d84.sum(1)+1e-9, axis=0).clip(lower=1e-9)
    out['hb_dept_entropy_84'] = (-(p_*np.log(p_)).sum(1)).values
    gp84 = tx[m84].groupby(['household_key','product_id']).sales_value.sum()
    mx = gp84.groupby('household_key').max().reindex(hh)
    tot = gp84.groupby('household_key').sum().reindex(hh)
    out['hb_top1prod_share_84'] = (mx/(tot+1e-9)).values
    told = tx[tx.day >= D-363]
    oldp = told[['household_key','product_id']].drop_duplicates()
    newr = t28.merge(oldp, on=['household_key','product_id'], how='left', indicator=True)
    newsp = newr[newr['_merge']=='left_only'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_new_prod_share_28'] = (newsp/(sp28h+1e-9)).values

    # D: stock-up basket shape
    b = tx.groupby(['household_key','basket_id']).agg(bspend=('sales_value','sum'), bday=('day','first'), bt=('trans_time','mean'))
    b84 = b[b.bday >= D-83]
    g = b84.groupby('household_key')
    bmax = g.bspend.max().reindex(hh)
    bmed = g.bspend.median().reindex(hh)
    bsum = g.bspend.sum().reindex(hh)
    out['su_max_over_med_84'] = (bmax/(bmed+1e-9)).values
    bidx = g.bspend.idxmax()
    bdaymap = b84.bday
    out['su_days_since_maxb_84'] = (D - bidx.map(bdaymap)).reindex(hh).fillna(999).values
    out['su_top1basket_share_84'] = (bmax/(bsum+1e-9)).values
    bb = b84.assign(bb=b84.bspend > 2*(bmed+1e-9))
    out['su_bigbasket_cnt_84'] = bb.groupby('household_key').bb.sum().reindex(hh).fillna(0).values
    out['su_stockup_share_84'] = bb.assign(sv=lambda x: x.bspend*x.bb.astype(float)).groupby('household_key').sv.sum().reindex(hh).fillna(0).values/(bsum+1e-9)
    q28 = t28.groupby('household_key').quantity.sum().reindex(hh)
    tr28 = t28.groupby('household_key').basket_id.nunique().reindex(hh)
    q84 = tx[m84].groupby('household_key').quantity.sum().reindex(hh)
    tr84 = tx[m84].groupby('household_key').basket_id.nunique().reindex(hh)
    out['su_qtyrate_28_84'] = ((q28/(tr28+1e-9))/((q84/(tr84+1e-9))+1e-9)).replace([np.inf,-np.inf],np.nan).values

    # J: time of day
    am = b84[b84.bt < 1200].groupby('household_key').bspend.sum().reindex(hh).fillna(0)
    ev = b84[b84.bt >= 1800].groupby('household_key').bspend.sum().reindex(hh).fillna(0)
    out['td_am_share_84'] = (am/(bsum+1e-9)).values
    out['td_eve_share_84'] = (ev/(bsum+1e-9)).values
    out['td_avg_time_84'] = g.bt.mean().reindex(hh).values

    # B: global level/trend/seasonal (scalars)
    tx2 = tx.assign(wk=(tx.day+8)//7)
    gw = tx2.groupby('wk').sales_value.sum()
    ga = tx2.groupby('wk').household_key.nunique()
    gwk = (gw/(ga+1e-9))
    cw = (D+8)//7
    def wmean(a, b_):
        sel = gwk.loc[a:b_]
        return float(sel.mean()) if len(sel) else np.nan
    g_level = wmean(max(1,cw-8), cw-1)
    g_prev = wmean(max(1,cw-16), cw-9)
    g_trend = np.nan
    if g_prev is not None and not np.isnan(g_prev):
        g_trend = g_level/(g_prev+1e-9)
    w0 = cw+1
    seas = np.nan
    seas_rel = 1.0
    if w0-52 >= 1:
        seas = wmean(w0-52, w0-49)
        if not np.isnan(seas) and not np.isnan(g_level):
            seas_rel = seas/(g_level+1e-9)
    n = len(hh)
    out['g_level'] = np.full(n, g_level if not np.isnan(g_level) else 0.0)
    out['g_trend'] = np.full(n, g_trend if not np.isnan(g_trend) else 1.0)
    out['g_season_rel'] = np.full(n, seas_rel)
    out['g_season_abs'] = np.full(n, seas)
    return out

df = agent_api.build_features(build_group)
print('built', df.shape, 'time', round(time.time()-t0))

e8 = agent_api.load_saved('e008_level_shape.parquet')
groups = {
 'candA_dm': ['dm_'], 'candB_g': ['g_'], 'candC_hb': ['hb_'], 'candD_su': ['su_'], 'candE_td': ['td_'],
}
for name, prefs in groups.items():
    cols = ['household_key','snapshot_day'] + [c for c in df.columns if any(c.startswith(p) for p in prefs)]
    sub = df[cols]
    mg = e8.merge(sub, on=['household_key','snapshot_day'], how='left')
    assert len(mg) == len(e8)
    p = agent_api.save_table(mg, name)
    print(name, mg.shape, 'nan%', round(mg[cols[2:]].isna().mean().mean()*100,1))
print('done', round(time.time()-t0), 's')


# ---- cell ----
import agent_api, numpy as np, pandas as pd, time
t0 = time.time()

def build_group(view, snapshot_day):
    D = int(snapshot_day)
    hh = view.households
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)

    prod = view.table('products')
    dep = prod[['product_id','department']].drop_duplicates('product_id')
    dep = dep.assign(department=dep.department.astype(str))
    tx = tx.merge(dep, on='product_id', how='left')
    tx['department'] = tx['department'].astype(object).where(tx['department'].notna(), 'UNK')

    m28 = tx.day >= D-27
    m84 = tx.day >= D-83

    # A: dept momentum
    s84 = tx[m84].groupby('department').sales_value.sum()
    topd = list(s84.sort_values(ascending=False).head(10).index)
    t28h = tx[m28].groupby(['household_key','department']).sales_value.sum()
    t84h = tx[m84].groupby(['household_key','department']).sales_value.sum()
    sh28 = t28h.unstack('department').reindex(columns=topd)
    sh84 = t84h.unstack('department').reindex(columns=topd)
    sh28 = sh28.div(sh28.sum(1)+1e-9, axis=0).reindex(hh)
    sh84 = sh84.div(sh84.sum(1)+1e-9, axis=0).reindex(hh)
    mom = (sh28.fillna(0) - sh84.fillna(0))
    for d in topd:
        out['dm_mom_'+d] = mom[d].values
    for d in topd[:5]:
        out['dm_sh28_'+d] = sh28[d].fillna(0).values

    # C: habit / repeat
    t28 = tx[m28]
    tprev = tx[(tx.day >= D-111) & (tx.day < D-27)]
    pairs = tprev[['household_key','product_id']].drop_duplicates()
    rep = t28.merge(pairs, on=['household_key','product_id'], how='left', indicator=True)
    rep28 = rep[rep['_merge']=='both'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    sp28h = t28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_repeat_share_28'] = (rep28/(sp28h+1e-9)).values
    n28 = t28.groupby('household_key').product_id.nunique().reindex(hh)
    nrep = rep[rep['_merge']=='both'].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['hb_rep_prod_frac_28'] = (nrep/(n28+1e-9)).values
    d84 = t84h.unstack('department').reindex(hh).fillna(0)
    p_ = d84.div(d84.sum(1)+1e-9, axis=0).clip(lower=1e-9)
    out['hb_dept_entropy_84'] = (-(p_*np.log(p_)).sum(1)).values
    gp84 = tx[m84].groupby(['household_key','product_id']).sales_value.sum()
    mx = gp84.groupby('household_key').max().reindex(hh)
    tot = gp84.groupby('household_key').sum().reindex(hh)
    out['hb_top1prod_share_84'] = (mx/(tot+1e-9)).values
    told = tx[tx.day >= D-363]
    oldp = told[['household_key','product_id']].drop_duplicates()
    newr = t28.merge(oldp, on=['household_key','product_id'], how='left', indicator=True)
    newsp = newr[newr['_merge']=='left_only'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_new_prod_share_28'] = (newsp/(sp28h+1e-9)).values

    # D: stock-up basket shape
    b = tx.groupby(['household_key','basket_id']).agg(bspend=('sales_value','sum'), bday=('day','first'), bt=('trans_time','mean'))
    b84 = b[b.bday >= D-83]
    g = b84.groupby('household_key')
    bmax = g.bspend.max().reindex(hh)
    bmed = g.bspend.median().reindex(hh)
    bsum = g.bspend.sum().reindex(hh)
    out['su_max_over_med_84'] = (bmax/(bmed+1e-9)).values
    bidx = g.bspend.idxmax()
    bdaymap = b84.bday
    out['su_days_since_maxb_84'] = (D - bidx.map(bdaymap)).reindex(hh).fillna(999).values
    out['su_top1basket_share_84'] = (bmax/(bsum+1e-9)).values
    thr = b84['household_key'].map(bmed)*2 + 1e-9
    bb = b84.assign(bb=(b84.bspend > thr).astype(float))
    out['su_bigbasket_cnt_84'] = bb.groupby('household_key').bb.sum().reindex(hh).fillna(0).values
    out['su_stockup_share_84'] = bb.assign(sv=bb.bspend*bb.bb).groupby('household_key').sv.sum().reindex(hh).fillna(0).values/(bsum+1e-9)
    q28 = t28.groupby('household_key').quantity.sum().reindex(hh)
    tr28 = t28.groupby('household_key').basket_id.nunique().reindex(hh)
    q84 = tx[m84].groupby('household_key').quantity.sum().reindex(hh)
    tr84 = tx[m84].groupby('household_key').basket_id.nunique().reindex(hh)
    out['su_qtyrate_28_84'] = ((q28/(tr28+1e-9))/((q84/(tr84+1e-9))+1e-9)).replace([np.inf,-np.inf],np.nan).values

    # J: time of day
    am = b84[b84.bt < 1200].groupby('household_key').bspend.sum().reindex(hh).fillna(0)
    ev = b84[b84.bt >= 1800].groupby('household_key').bspend.sum().reindex(hh).fillna(0)
    out['td_am_share_84'] = (am/(bsum+1e-9)).values
    out['td_eve_share_84'] = (ev/(bsum+1e-9)).values
    out['td_avg_time_84'] = g.bt.mean().reindex(hh).values

    # B: global level/trend/seasonal (scalars)
    tx2 = tx.assign(wk=(tx.day+8)//7)
    gw = tx2.groupby('wk').sales_value.sum()
    ga = tx2.groupby('wk').household_key.nunique()
    gwk = (gw/(ga+1e-9))
    cw = (D+8)//7
    def wmean(a, b_):
        sel = gwk.loc[a:b_]
        return float(sel.mean()) if len(sel) else np.nan
    g_level = wmean(max(1,cw-8), cw-1)
    g_prev = wmean(max(1,cw-16), cw-9)
    g_trend = np.nan
    if g_prev is not None and not np.isnan(g_prev):
        g_trend = g_level/(g_prev+1e-9)
    w0 = cw+1
    seas = np.nan
    seas_rel = 1.0
    if w0-52 >= 1:
        seas = wmean(w0-52, w0-49)
        if not np.isnan(seas) and not np.isnan(g_level):
            seas_rel = seas/(g_level+1e-9)
    n = len(hh)
    out['g_level'] = np.full(n, g_level if not np.isnan(g_level) else 0.0)
    out['g_trend'] = np.full(n, g_trend if not np.isnan(g_trend) else 1.0)
    out['g_season_rel'] = np.full(n, seas_rel)
    out['g_season_abs'] = np.full(n, seas)
    return out

df = agent_api.build_features(build_group)
print('built', df.shape, 'time', round(time.time()-t0))

e8 = agent_api.load_saved('e008_level_shape.parquet')
groups = {
 'candA_dm': ['dm_'], 'candB_g': ['g_'], 'candC_hb': ['hb_'], 'candD_su': ['su_'], 'candE_td': ['td_'],
}
for name, prefs in groups.items():
    cols = ['household_key','snapshot_day'] + [c for c in df.columns if any(c.startswith(p) for p in prefs)]
    sub = df[cols]
    mg = e8.merge(sub, on=['household_key','snapshot_day'], how='left')
    assert len(mg) == len(e8)
    p = agent_api.save_table(mg, name)
    print(name, mg.shape, 'nan%', round(mg[cols[2:]].isna().mean().mean()*100,1))
print('done', round(time.time()-t0), 's')


# ---- cell ----
import agent_api, numpy as np, pandas as pd, time
t0 = time.time()

def build_group(view, snapshot_day):
    D = int(snapshot_day)
    hh = view.households
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)

    prod = view.table('products')
    dep = prod[['product_id','department']].drop_duplicates('product_id')
    dep = dep.assign(department=dep.department.astype(str))
    tx = tx.merge(dep, on='product_id', how='left')
    tx['department'] = tx['department'].astype(object).where(tx['department'].notna(), 'UNK')

    m28 = tx.day >= D-27
    m84 = tx.day >= D-83

    # A: dept momentum
    s84 = tx[m84].groupby('department').sales_value.sum()
    topd = list(s84.sort_values(ascending=False).head(10).index)
    t28h = tx[m28].groupby(['household_key','department']).sales_value.sum()
    t84h = tx[m84].groupby(['household_key','department']).sales_value.sum()
    sh28 = t28h.unstack('department').reindex(columns=topd)
    sh84 = t84h.unstack('department').reindex(columns=topd)
    sh28 = sh28.div(sh28.sum(1)+1e-9, axis=0).reindex(hh)
    sh84 = sh84.div(sh84.sum(1)+1e-9, axis=0).reindex(hh)
    mom = (sh28.fillna(0) - sh84.fillna(0))
    for d in topd:
        out['dm_mom_'+d] = mom[d].values
    for d in topd[:5]:
        out['dm_sh28_'+d] = sh28[d].fillna(0).values

    # C: habit / repeat
    t28 = tx[m28]
    tprev = tx[(tx.day >= D-111) & (tx.day < D-27)]
    pairs = tprev[['household_key','product_id']].drop_duplicates()
    rep = t28.merge(pairs, on=['household_key','product_id'], how='left', indicator=True)
    rep28 = rep[rep['_merge']=='both'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    sp28h = t28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_repeat_share_28'] = (rep28/(sp28h+1e-9)).values
    n28 = t28.groupby('household_key').product_id.nunique().reindex(hh)
    nrep = rep[rep['_merge']=='both'].groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['hb_rep_prod_frac_28'] = (nrep/(n28+1e-9)).values
    d84 = t84h.unstack('department').reindex(hh).fillna(0)
    p_ = d84.div(d84.sum(1)+1e-9, axis=0).clip(lower=1e-9)
    out['hb_dept_entropy_84'] = (-(p_*np.log(p_)).sum(1)).values
    gp84 = tx[m84].groupby(['household_key','product_id']).sales_value.sum()
    mx = gp84.groupby('household_key').max().reindex(hh)
    tot = gp84.groupby('household_key').sum().reindex(hh)
    out['hb_top1prod_share_84'] = (mx/(tot+1e-9)).values
    told = tx[tx.day >= D-363]
    oldp = told[['household_key','product_id']].drop_duplicates()
    newr = t28.merge(oldp, on=['household_key','product_id'], how='left', indicator=True)
    newsp = newr[newr['_merge']=='left_only'].groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['hb_new_prod_share_28'] = (newsp/(sp28h+1e-9)).values

    # D: stock-up basket shape
    b = tx.groupby(['household_key','basket_id']).agg(bspend=('sales_value','sum'), bday=('day','first'), bt=('trans_time','mean'))
    b84 = b[b.bday >= D-83].copy()
    g = b84.groupby(level=0)
    bmax = g.bspend.max().reindex(hh)
    bmed = g.bspend.median().reindex(hh)
    bsum = g.bspend.sum().reindex(hh)
    out['su_max_over_med_84'] = (bmax/(bmed+1e-9)).values
    bidx = g.bspend.idxmax()
    bdaymap = b84.bday
    out['su_days_since_maxb_84'] = (D - bidx.map(bdaymap)).reindex(hh).fillna(999).values
    out['su_top1basket_share_84'] = (bmax/(bsum+1e-9)).values
    hhkey = pd.Series(b84.index.get_level_values(0), index=b84.index)
    thr = hhkey.map(bmed)*2 + 1e-9
    b84['bb'] = (b84.bspend > thr).astype(float)
    b84['sv'] = b84.bspend * b84.bb
    out['su_bigbasket_cnt_84'] = b84.groupby(level=0).bb.sum().reindex(hh).fillna(0).values
    out['su_stockup_share_84'] = b84.groupby(level=0).sv.sum().reindex(hh).fillna(0).values/(bsum+1e-9)
    q28 = t28.groupby('household_key').quantity.sum().reindex(hh)
    tr28 = t28.groupby('household_key').basket_id.nunique().reindex(hh)
    q84 = tx[m84].groupby('household_key').quantity.sum().reindex(hh)
    tr84 = tx[m84].groupby('household_key').basket_id.nunique().reindex(hh)
    out['su_qtyrate_28_84'] = ((q28/(tr28+1e-9))/((q84/(tr84+1e-9))+1e-9)).replace([np.inf,-np.inf],np.nan).values

    # J: time of day
    am = b84[b84.bt < 1200].groupby(level=0).bspend.sum().reindex(hh).fillna(0)
    ev = b84[b84.bt >= 1800].groupby(level=0).bspend.sum().reindex(hh).fillna(0)
    out['td_am_share_84'] = (am/(bsum+1e-9)).values
    out['td_eve_share_84'] = (ev/(bsum+1e-9)).values
    out['td_avg_time_84'] = g.bt.mean().reindex(hh).values

    # B: global level/trend/seasonal (scalars)
    tx2 = tx.assign(wk=(tx.day+8)//7)
    gw = tx2.groupby('wk').sales_value.sum()
    ga = tx2.groupby('wk').household_key.nunique()
    gwk = (gw/(ga+1e-9))
    cw = (D+8)//7
    def wmean(a, b_):
        sel = gwk.loc[a:b_]
        return float(sel.mean()) if len(sel) else np.nan
    g_level = wmean(max(1,cw-8), cw-1)
    g_prev = wmean(max(1,cw-16), cw-9)
    g_trend = np.nan
    if g_prev is not None and not np.isnan(g_prev):
        g_trend = g_level/(g_prev+1e-9)
    w0 = cw+1
    seas = np.nan
    seas_rel = 1.0
    if w0-52 >= 1:
        seas = wmean(w0-52, w0-49)
        if not np.isnan(seas) and not np.isnan(g_level):
            seas_rel = seas/(g_level+1e-9)
    n = len(hh)
    out['g_level'] = np.full(n, g_level if not np.isnan(g_level) else 0.0)
    out['g_trend'] = np.full(n, g_trend if not np.isnan(g_trend) else 1.0)
    out['g_season_rel'] = np.full(n, seas_rel)
    out['g_season_abs'] = np.full(n, seas)
    return out

df = agent_api.build_features(build_group)
print('built', df.shape, 'time', round(time.time()-t0))

e8 = agent_api.load_saved('e008_level_shape.parquet')
groups = {
 'candA_dm': ['dm_'], 'candB_g': ['g_'], 'candC_hb': ['hb_'], 'candD_su': ['su_'], 'candE_td': ['td_'],
}
for name, prefs in groups.items():
    cols = ['household_key','snapshot_day'] + [c for c in df.columns if any(c.startswith(p) for p in prefs)]
    sub = df[cols]
    mg = e8.merge(sub, on=['household_key','snapshot_day'], how='left')
    assert len(mg) == len(e8)
    p = agent_api.save_table(mg, name)
    print(name, mg.shape, 'nan%', round(mg[cols[2:]].isna().mean().mean()*100,1))
print('done', round(time.time()-t0), 's')


# ---- cell ----
import agent_api, numpy as np, pandas as pd, time
t0=time.time()
tt = agent_api.train_targets()

def prep(path):
    t = agent_api.load_saved(path)
    feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    m = t.merge(tt, on=['household_key','snapshot_day'], how='left')
    isval = m.future_spend_4w.isna().values
    Xdf = m[feat].copy()
    for c in feat:
        if str(Xdf[c].dtype) in ('object','category','bool'):
            Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            Xdf[c] = Xdf[c].astype(float)
    return feat, Xdf.values, isval

def binize(Xref, Xnew, nbins=32):
    p = Xref.shape[1]
    edges = []; Bref = np.zeros(Xref.shape, np.int16); Bnew = np.zeros(Xnew.shape, np.int16)
    for j in range(p):
        col = Xref[:,j]; med = np.nanmedian(col)
        if np.isnan(med): med = 0.0
        colf = np.where(np.isnan(col), med, col)
        qs = np.unique(np.quantile(colf, np.linspace(0,1,nbins+1)[1:-1]))
        edges.append(qs)
        Bref[:,j] = np.searchsorted(qs, colf, 'right')
        coln = Xnew[:,j]; coln = np.where(np.isnan(coln), med, coln)
        Bnew[:,j] = np.searchsorted(qs, coln, 'right')
    return Bref, Bnew, edges

def gbm_fit(Btr, ytr, edges, depth=4, T=100, lr=0.1, lam=1.0, minleaf=20, seed=0, sub=0.85):
    rng = np.random.RandomState(seed); n, p = Btr.shape
    nb = np.array([len(e) for e in edges]); off = np.concatenate([[0], np.cumsum(nb+1)[:-1]])
    total = int(off[-1] + nb[-1] + 1); F = np.zeros(n); trees = []
    for it in range(T):
        resid = ytr - F; rows0 = rng.choice(n, int(n*sub), replace=False)
        leafval = {}; splits = {}; queue = [(rows0, 0, 0)]
        while queue:
            rows, d, nid = queue.pop()
            g = resid[rows]; G = g.sum(); H = len(rows)
            if d >= depth or H < 2*minleaf:
                leafval[nid] = lr*(G/(H+lam)); F[rows] += leafval[nid]; continue
            subM = Btr[rows]; flat = (subM.astype(np.int64)+off[None,:]).ravel()
            hist = np.bincount(flat, weights=np.repeat(g,p), minlength=total)
            cnt = np.bincount(flat, minlength=total)
            bg, bj, bb = 0.0, -1, -1
            for j in range(p):
                h = hist[off[j]:off[j]+nb[j]+1]; c = cnt[off[j]:off[j]+nb[j]+1]
                if len(h) < 2: continue
                GL = np.cumsum(h)[:-1]; HL = np.cumsum(c)[:-1]; GR = G-GL; HR = H-HL
                gain = GL*GL/(HL+lam)+GR*GR/(HR+lam)-G*G/(H+lam)
                gain[(HL<minleaf)|(HR<minleaf)] = -1
                b = int(np.argmax(gain))
                if gain[b] > bg: bg, bj, bb = gain[b], j, b
            if bj < 0:
                leafval[nid] = lr*(G/(H+lam)); F[rows] += leafval[nid]; continue
            mask = subM[:, bj] <= bb; lid, rid = nid*2+1, nid*2+2
            splits[nid] = (bj, bb, lid, rid)
            queue.append((rows[mask], d+1, lid)); queue.append((rows[~mask], d+1, rid))
        trees.append((splits, leafval))
    return trees

def gbm_predict(Bev, trees):
    out = np.zeros(Bev.shape[0])
    for splits, leafval in trees:
        active = {0: np.arange(Bev.shape[0])}
        while active:
            nid, rows = active.popitem()
            if nid in leafval: out[rows] += leafval[nid]; continue
            j, b, lid, rid = splits[nid]
            msk = Bev[rows, j] <= b
            if msk.any(): active[lid] = np.concatenate([active.get(lid, np.array([],int)), rows[msk]])
            if (~msk).any(): active[rid] = np.concatenate([active.get(rid, np.array([],int)), rows[~msk]])
    return out

def cv(path, tag):
    feat, X, isval = prep(path)
    sd = None
    t = agent_api.load_saved(path); sdv = t.snapshot_day.values.astype(int)
    y = None
    m = t.merge(tt, on=['household_key','snapshot_day'], how='left')
    yv = m.future_spend_4w.values
    maes = []
    for tr_max, evs in [(347,[375]), (375,[403,431])]:
        tr = sdv <= tr_max; ev = np.isin(sdv, evs)
        Btr, Bev, edges = binize(X[tr], X[ev])
        trees = gbm_fit(Btr, yv[tr].astype(float), edges)
        pr = gbm_predict(Bev, trees)
        maes.append(np.abs(pr - yv[ev].astype(float)).mean())
    print(f'{tag:14s} MAE {maes[0]:.2f} / {maes[1]:.2f}  AVG {np.mean(maes):.3f}  (p={X.shape[1]})')
    return np.mean(maes)

base = cv('e008_level_shape.parquet', 'E008')
for name, tag in [('candA_dm','+deptMom'),('candB_g','+global'),('candC_hb','+habit'),('candD_su','+stockup'),('candE_td','+timeofday')]:
    cv(name+'.parquet', tag)
# all combined
e8 = agent_api.load_saved('e008_level_shape.parquet')
dfall = e8
for name in ['candA_dm','candB_g','candC_hb','candD_su','candE_td']:
    t = agent_api.load_saved(name+'.parquet')
    newc = [c for c in t.columns if c not in dfall.columns]
    dfall = dfall.merge(t[['household_key','snapshot_day']+newc], on=['household_key','snapshot_day'], how='left')
p = agent_api.save_table(dfall, 'candAll')
cv('candAll.parquet', '+ALL')
print('done', round(time.time()-t0), 's')


# ---- cell ----
import agent_api, pandas as pd
e8 = agent_api.load_saved('e008_level_shape.parquet')
m = e8
for name in ['candB_g','candD_su','candE_td']:
    t = agent_api.load_saved(name+'.parquet')
    newc = [c for c in t.columns if c not in m.columns]
    m = m.merge(t[['household_key','snapshot_day']+newc], on=['household_key','snapshot_day'], how='left')
assert len(m)==len(e8)
p = agent_api.save_table(m, 'candGST')
print('candGST', m.shape, list(m.columns)[-11:])
