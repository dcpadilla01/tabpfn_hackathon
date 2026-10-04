import pandas as pd, numpy as np
try:
    _a = agent_api
except NameError:
    import agent_api as _a

t = _a.load_saved('e003_catmix.parquet')
print('e003 shape', t.shape)
print('e003 cols:', list(t.columns))

tt = _a.train_targets()
y = tt['future_spend_4w']
print('\ntargets', tt.shape)
print(y.describe())
print('zero share', float((y==0).mean()))
print(y.quantile([.05,.1,.25,.5,.75,.9,.95,.99]).to_dict())

v = _a.snapshot()
print('\nview day/week:', v.day, v.week)
print('households type', type(v.households), 'len', len(v.households))
tx = v.transactions
print('tx shape', tx.shape)
print(tx[['sales_value','quantity','retail_disc','coupon_disc','coupon_match_disc']].describe())
print('neg sales share', float((tx.sales_value<0).mean()), 'qty<=0 share', float((tx.quantity<=0).mean()))
print('hh with tx', tx.household_key.nunique())

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'])
print('merged', df.shape)
y = df['future_spend_4w']

def mae(p): return float(np.mean(np.abs(p-y)))

print('median pred MAE:', mae(np.full(len(y), y.median())))
print('mean pred MAE:', mae(np.full(len(y), y.mean())))
print('spend_l1 MAE:', mae(df.spend_l1))
print('mean(l1..l3) MAE:', mae(df[['spend_l1','spend_l2','spend_l3']].mean(1)))
print('spend_l123_mean MAE:', mae(df.spend_l123_mean))
print('spend_rate28 MAE:', mae(df.spend_rate28))
print('spend_l13 MAE:', mae(df.spend_l13))
# shrunk versions
for w in [0.5,0.6,0.7,0.8]:
    print(f'shrink {w}: l1*w MAE:', mae(df.spend_l1*w))
# correlation structure
print('\ncorr with y:')
for c in ['spend_l1','spend_l2','spend_l3','spend_l123_mean','spend_l13','days_since_last','zero_recent','momentum','spend_rate28']:
    print(c, round(float(df[c].corr(y)),3), '| log-log', round(float(np.corrcoef(np.log1p(df[c].clip(lower=0)), np.log1p(y))[0,1]),3))
# zero structure
print('\nmean y by spend_l1 bucket:')
df['b'] = pd.cut(df.spend_l1, [-1,0,25,75,150,300,10000])
print(df.groupby('b', observed=True).agg(n=('y','size'), ymean=('y','mean'), l1mean=('spend_l1','mean'), ymed=('y','median')))
print('\nmean y by days_since_last:')
df['b2'] = pd.cut(df.days_since_last, [-1,3,7,14,21,28,56,10000])
print(df.groupby('b2', observed=True).agg(n=('y','size'), ymean=('y','mean'), pzero=('y', lambda s:(s==0).mean())))

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'])
y = df['future_spend_4w'].values
def mae(p): return float(np.mean(np.abs(p-y)))

# hand predictors on train rows
p1 = df[['spend_l1','spend_l2','spend_l3']].mean(1).values
p2 = df.spend_rate28.values
print('mean(l1..l3):', round(mae(p1),2), ' rate28:', round(mae(p2),2))
for a in [0.3,0.5,0.7]:
    print(f'blend {a}*p1+{1-a}*p2:', round(mae(a*p1+(1-a)*p2),2))

# per-snapshot target stats (seasonality)
g = df.groupby('snapshot_day').agg(n=('future_spend_4w','size'), mean=('future_spend_4w','mean'), med=('future_spend_4w','median'), p1m=('spend_l1','mean'))
print('\nper snapshot:'); print(g.round(1))

# residual of naive predictor by snapshot
df['res'] = np.abs(df[['spend_l1','spend_l2','spend_l3']].mean(1).values - y)
print('\nnaive MAE per snapshot:'); print(df.groupby('snapshot_day')['res'].mean().round(1))

# zero structure: mean y when l1==0 vs >0
print('\nl1==0: n=', int((df.spend_l1==0).sum()), 'mean y=', round(float(y[df.spend_l1==0].mean()),1))
print('l1>0: mean y=', round(float(y[df.spend_l1>0].mean()),1))
print('l1==0 & l2==0 & l3==0: n=', int(((df.spend_l1==0)&(df.spend_l2==0)&(df.spend_l3==0)).sum()),
      'mean y=', round(float(y[(df.spend_l1==0)&(df.spend_l2==0)&(df.spend_l3==0)].mean()),1))

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
tr_days = A.snapshot_days()
print(tr_days)
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
df = t.merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day.isin(tr_days['train'])]
va = df[df.snapshot_day.isin(tr_days['validation'])]
print('train rows', len(tr), 'val rows', len(va))

def ridge_fit(X, y, lam):
    n, p = X.shape
    D = np.concatenate([np.zeros((1,1)), np.ones((1,p))], axis=1)  # not used
    A_ = np.hstack([np.ones((n,1)), X])
    R = A_.T @ A_ + lam*np.eye(p+1); R[0,0] -= lam  # don't penalize intercept
    return np.linalg.solve(R, A_.T @ y)

def prep(Xtr, Xva, mu=None, sd=None):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    return (Xtr-mu)/sd, (Xva-mu)/sd

Xtr_raw = tr[feats].fillna(0).values.astype(float)
Xva_raw = va[feats].fillna(0).values.astype(float)
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values

# lambda selection via leave-one-snapshot-out on train
lams = [1,3,10,30,100,300,1000,3000]
Xtr_s, _ = prep(Xtr_raw, Xtr_raw)
cv = {}
for lam in lams:
    errs = []
    for d in tr_days['train']:
        m = tr.snapshot_day.values != d
        Xi, yi = Xtr_s[m], ytr[m]
        mu = Xi.mean(0); sd = Xi.std(0); sd[sd==0]=1
        w = ridge_fit((Xi-mu)/sd, yi, lam)
        Xo = Xtr_s[~m]; Xo = (Xo-mu)/sd
        errs.append(np.mean(np.abs(Xo @ w[1:] + w[0] - ytr[~m])))
    cv[lam] = float(np.mean(errs))
print('LOSO-CV MAE by lambda:', {k: round(v,2) for k,v in cv.items()})
lam_best = min(cv, key=cv.get)
print('best lam', lam_best)

Xtr_s, Xva_s = prep(Xtr_raw, Xva_raw)
w = ridge_fit(Xtr_s, ytr, lam_best)
pv = Xva_s @ w[1:] + w[0]
print('LOCAL ridge on E003 feats: val MAE', round(float(np.mean(np.abs(pv-yva))),3), '(harness E003: 63.050)')
ptr = Xtr_s @ w[1:] + w[0]
print('train MAE', round(float(np.mean(np.abs(ptr-ytr))),3))
# clip at 0
print('val MAE clipped>=0:', round(float(np.mean(np.abs(np.clip(pv,0,None)-yva))),3))

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

tr_days = A.snapshot_days()['train']
t3 = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
base_feats = [c for c in t3.columns if c not in ('household_key','snapshot_day')]

def feats_for(tx, s):
    w = ((s - tx['day']) // 28).astype('int64')
    tx = tx.assign(w=w)
    g = tx.groupby(['household_key','w'])
    spend = g['sales_value'].sum().unstack()
    first_day = tx.groupby('household_key')['day'].min()
    elig = first_day.index[first_day <= s-84]
    spend = spend.reindex(index=elig, columns=list(range(13))).fillna(0.0)
    S = spend.values; n = len(elig)
    act = (S > 0).astype(float)
    f = pd.DataFrame(index=elig)
    # A: recency-decayed spend
    for hl, nm in [(2,'hl2'), (4,'hl4')]:
        wt = 0.5 ** (np.arange(6) / hl); wt = wt / wt.sum()
        f['dec_' + nm] = (S[:, :6] * wt).sum(1)
    # B: activity/level decomposition
    f['act_rate_3'] = act[:, :3].mean(1)
    f['act_rate_6'] = act[:, :6].mean(1)
    f['act_rate_13'] = act[:, :13].mean(1)
    act13 = act[:, :13].sum(1)
    mean_active_13 = np.where(act13 > 0, S[:, :13].sum(1) / np.maximum(act13, 1), 0.0)
    f['mean_active_13'] = mean_active_13
    f['expected_13'] = f['act_rate_13'] * mean_active_13
    a6 = act[:, :6].sum(1)
    f['expected_6'] = f['act_rate_6'] * np.where(a6 > 0, S[:, :6].sum(1) / np.maximum(a6, 1), 0.0)
    nwin = np.maximum(((s - first_day.loc[elig] + 1) // 28).values, 1)
    f['spend_per_win'] = S.sum(1) / nwin
    # C: dormancy gaps / resumption
    zrun = np.ones(n); zs = np.zeros(n)
    for k in range(13):
        zrun = zrun * (act[:, k] == 0); zs += zrun
    f['zero_streak'] = zs
    rs = np.zeros(n); rc = np.zeros(n)
    for k in range(1, 13):
        m = (act[:, k] == 0) & (act[:, k-1] > 0)
        rs[m] += S[m, k-1]; rc[m] += 1
    f['mean_resumption'] = np.where(rc > 0, rs / np.maximum(rc, 1), mean_active_13)
    # D: market seasonal anchor (target window last year = window w12)
    mkt_tot = S.sum(0); mkt_nact = act.sum(0)
    mkt_pa = np.where(mkt_nact > 0, mkt_tot / np.maximum(mkt_nact, 1), np.nan)
    f['mkt_tot_w12'] = mkt_tot[12]; f['mkt_nact_w12'] = mkt_nact[12]
    f['mkt_peract_w12'] = mkt_pa[12]; f['mkt_peract_w0'] = mkt_pa[0]
    r = mkt_pa[12] / mkt_pa[0] if mkt_pa[0] and mkt_pa[0] > 0 else np.nan
    f['mkt_ratio'] = r
    naive = S[:, :3].mean(1)
    f['naive_x_mkt'] = naive * r
    f['exp13_x_mkt'] = f['expected_13'] * r
    # E: household seasonal multiplier
    f['hh_seas_mult'] = S[:, 12] / np.maximum(f['spend_per_win'], 1e-9)
    # F: target-window annual phase
    ph = 2 * np.pi * ((s + 14) % 364) / 364
    f['sin_ann'] = np.sin(ph); f['cos_ann'] = np.cos(ph)
    return f

new = {}
for s in tr_days:
    v = A.snapshot(as_of_day=s)
    new[s] = feats_for(v.transactions, s).assign(snapshot_day=s)
newdf = pd.concat(new.values()).reset_index()
df = t3.merge(tt, on=['household_key','snapshot_day']).merge(newdf, on=['household_key','snapshot_day'], how='left')
print('merged', df.shape)

groups = {
 'A_recdecay': ['dec_hl2','dec_hl4'],
 'B_actlevel': ['act_rate_3','act_rate_6','act_rate_13','mean_active_13','expected_13','expected_6','spend_per_win'],
 'C_gaps': ['zero_streak','mean_resumption'],
 'D_mkt': ['mkt_tot_w12','mkt_nact_w12','mkt_peract_w12','mkt_peract_w0','mkt_ratio','naive_x_mkt','exp13_x_mkt'],
 'E_hhseas': ['hh_seas_mult'],
 'F_phase': ['sin_ann','cos_ann'],
}

def loso(cols, lams=(3,30,300)):
    Xall = df[cols].astype(float)
    mu = Xall.mean(); sd = Xall.std(); sd[sd==0]=1
    Xs = ((Xall - mu) / sd).fillna(0.0).values
    y = df['future_spend_4w'].values
    days = df['snapshot_day'].values
    best = (1e9, None)
    for lam in lams:
        errs = []
        for d in tr_days:
            m = days != d
            Xi, yi = Xs[m], y[m]
            A_ = np.hstack([np.ones((len(Xi),1)), Xi])
            R = A_.T @ A_ + lam*np.eye(Xi.shape[1]+1); R[0,0] -= lam
            w = np.linalg.solve(R, A_.T @ yi)
            Xo = np.hstack([np.ones((len(Xs[~m]),1)), Xs[~m]])
            errs.append(np.mean(np.abs(Xo @ w - y[~m])))
        e = float(np.mean(errs))
        if e < best[0]: best = (e, lam)
    return best

res = {}
base_only = loso(base_feats); res['base(E003)'] = base_only
print('base(E003):', round(base_only[0],2), 'lam', base_only[1])
acc = []
for gname, gcols in groups.items():
    acc += gcols
    r = loso(base_feats + acc)
    res['+'+gname] = r
    print(f'+{gname}: {round(r[0],2)} (lam {r[1]})  cum_feats={len(base_feats)+len(acc)}')
r = loso(base_feats + [c for g in groups.values() for c in g])
print('ALL:', round(r[0],2), 'lam', r[1])
# market-only and actlevel-only
print('+D only:', round(loso(base_feats + groups['D_mkt'])[0],2))
print('+B only:', round(loso(base_feats + groups['B_actlevel'])[0],2))

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

tr_days = A.snapshot_days()['train']
b = A.baseline_features()
tt = A.train_targets()
bfeats = [c for c in b.columns if c not in ('household_key','snapshot_day')]
print('E000 feats:', bfeats)
db = b.merge(tt, on=['household_key','snapshot_day'])
X = db[bfeats]
# dummies for categorical
Xn = pd.get_dummies(X.astype(object).where(~X.isna(), X), dummy_na=True)
Xn = Xn.astype(float)
y = db.future_spend_4w.values; days = db.snapshot_day.values
mu = Xn.mean(); sd = Xn.std(); sd[sd==0]=1
Xs = ((Xn-mu)/sd).fillna(0).values
errs=[]
for d in tr_days:
    m = days!=d
    A_ = np.hstack([np.ones((m.sum(),1)), Xs[m]])
    R = A_.T@A_ + 30*np.eye(A_.shape[1]); R[0,0]-=30
    w = np.linalg.solve(R, A_.T@y[m])
    Xo = np.hstack([np.ones(((~m).sum(),1)), Xs[~m]])
    errs.append(np.mean(np.abs(Xo@w - y[~m])))
print('E000 ridge LOSO MAE:', round(float(np.mean(errs)),2), '(harness E000: 92.446)')

# conditional median structure
t3 = A.load_saved('e003_catmix.parquet')
df = t3.merge(tt, on=['household_key','snapshot_day'])
df['naive'] = df[['spend_l1','spend_l2','spend_l3']].mean(1)
tr = df[df.snapshot_day.isin(tr_days)]
va = df[~df.snapshot_day.isin(tr_days)]
print('\nval rows:', len(va), 'val snap days:', sorted(va.snapshot_day.unique()))

def pava(score, y, w=None):
    # isotonic regression (weighted, increasing)
    n=len(score); w=np.ones(n) if w is None else w
    o=np.argsort(score, kind='stable'); xs=score[o]; ys=y[o].astype(float); ws=w[o]
    # pool adjacent violators
    vy=[]; vw=[]; 
    for i in range(n):
        vy.append(ys[i]); vw.append(ws[i])
        while len(vy)>1 and vy[-2]>vy[-1]:
            v2=(vy[-2]*vw[-2]+vy[-1]*vw[-1]); w2=vw[-2]+vw[-1]
            vy[-2:]=[]; vw[-2:]=[]
            vy.append(v2); vw.append(w2)
    out=np.empty(n)
    idx=0
    # expand blocks
    blocks=[]
    cnt=0
    for v,ww in zip(vy,vw):
        k=int(round(ww)) if np.allclose(ws,1) else None
        blocks.append((v,ww))
    # reconstruct via cumulative weights
    res=np.empty(n); pos=0
    for v,ww in zip(vy,vw):
        k=int(np.ceil(ww-1e-9))
        res[pos:pos+k]=v; pos+=k
    out[o]=res
    return out

def eval_pred(pred, name):
    print(f'{name}: train MAE {np.mean(np.abs(pred-y[df.index.isin(tr.index)])):.2f}' if False else f'{name}: val MAE {np.mean(np.abs(pred - va.future_spend_4w.values)):.2f}')

# Variant 1: isotonic on naive (fit train, apply val)
sc_tr = tr.naive.values; ytr=tr.future_spend_4w.values
fit = pava(np.sort(sc_tr), ytr[np.argsort(sc_tr, kind='stable')])  # placeholder
iso = pava(sc_tr, ytr)
# build step function from (sorted unique score, iso values)
o=np.argsort(sc_tr, kind='stable'); ss=sc_tr[o]; vv=iso[o]
# compress to knots
knot_x=[]; knot_y=[]
i=0
while i < len(ss):
    j=i
    while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
    knot_x.append(ss[i]); knot_y.append(vv[j]); i=j+1
knot_x=np.array(knot_x); knot_y=np.array(knot_y)
def iso_apply(s):
    return np.interp(s, knot_x, knot_y)
pred_tr = iso_apply(tr.naive.values); pred_va = iso_apply(va.naive.values)
print('V1 isotonic(naive): train MAE', round(float(np.mean(np.abs(pred_tr-ytr))),2),
      'val MAE', round(float(np.mean(np.abs(pred_va-va.future_spend_4w.values))),2))
print('naive raw val MAE:', round(float(np.mean(np.abs(va.naive.values-va.future_spend_4w.values))),2))

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

class GBM:
    def __init__(self, rounds=300, lr=0.1, depth=4, lam=1.0, min_leaf=20, bins=32, seed=0):
        self.p = dict(rounds=rounds, lr=lr, depth=depth, lam=lam, min_leaf=min_leaf, bins=bins, seed=seed)
    def fit(self, X, y):
        p = self.p; n, m = X.shape
        self.edges = []
        Xb = np.zeros((n, m), dtype=np.int8)
        for j in range(m):
            col = X[:, j]
            qs = np.quantile(col, np.linspace(0, 1, p['bins']+1))
            edges = np.unique(qs)
            if len(edges) < 2: edges = np.array([col.min()-1, col.max()+1])
            b = np.clip(np.searchsorted(edges[1:-1], col, side='right'), 0, len(edges)-2)
            Xb[:, j] = b; self.edges.append(edges)
        pred = np.full(n, y.mean()); self.trees = []
        for t in range(p['rounds']):
            g = pred - y
            nodes = [(np.arange(n), 0)]; tree = []
            while nodes:
                idx, d = nodes.pop(0)
                G = g[idx].sum(); H = len(idx)
                if d >= p['depth'] or H < 2*p['min_leaf'] or np.abs(g[idx]).max() < 1e-12:
                    tree.append(('L', -G/(H+p['lam']))); continue
                best = (None, -1e18, None)
                for j in range(m):
                    b = Xb[idx, j]
                    hg = np.bincount(b, weights=g[idx], minlength=p['bins'])[:p['bins']]
                    hn = np.bincount(b, minlength=p['bins'])[:p['bins']].astype(float)
                    cg = np.cumsum(hg); cn = np.cumsum(hn)
                    gl = cg[:-1]; nl = cn[:-1]; gr = G-gl; nr = H-nl
                    valid = (nl >= p['min_leaf']) & (nr >= p['min_leaf'])
                    if not valid.any(): continue
                    score = np.where(valid, gl**2/(nl+p['lam']) + gr**2/(nr+p['lam']), -1e18)
                    k = int(np.argmax(score))
                    if score[k] > best[1]:
                        thr = self.edges[j][k+1]
                        best = (j, score[k], thr)
                j, sc, thr = best
                if j is None:
                    tree.append(('L', -G/(H+p['lam']))); continue
                mask = X[idx, j] <= thr
                tree.append(('S', j, thr))
                nodes.append((idx[mask], d+1)); nodes.append((idx[~mask], d+1))
            cur = np.zeros(n, dtype=int); upd = np.zeros(n)
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    upd[cur == i] = node[1]
            pred = pred + p['lr'] * upd
            self.trees.append(tree)
        self.base = y.mean()
        return self
    def predict(self, X):
        out = np.full(len(X), self.base)
        for tree in self.trees:
            cur = np.zeros(len(X), dtype=int); leaf = np.zeros(len(X))
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    leaf[cur == i] = node[1]
            out += self.p['lr'] * leaf
        return out

def loso_gbm(df, feats, rounds=200, depth=4, lr=0.1):
    tr_days = sorted(df.snapshot_day.unique())
    X = df[feats].astype(float).values
    y = df.future_spend_4w.values; days = df.snapshot_day.values
    errs = []
    for d in tr_days:
        m = days != d
        mdl = GBM(rounds=rounds, depth=depth, lr=lr).fit(X[m], y[m])
        errs.append(float(np.mean(np.abs(mdl.predict(X[~m]) - y[~m]))))
    return float(np.mean(errs))

b = A.baseline_features(); tt = A.train_targets()
bfeats = [c for c in b.columns if c not in ('household_key','snapshot_day')]
db = b.merge(tt, on=['household_key','snapshot_day'])
for c in bfeats:
    if db[c].dtype == object: db[c] = db[c].astype('category').cat.codes.replace(-1, np.nan)
db[bfeats] = db[bfeats].astype(float)
print('E000 GBM LOSO:', round(loso_gbm(db, bfeats),2), '(harness 92.446)', flush=True)

t3 = A.load_saved('e003_catmix.parquet')
df3 = t3.merge(tt, on=['household_key','snapshot_day'])
f3 = [c for c in t3.columns if c not in ('household_key','snapshot_day')]
print('E003 GBM LOSO:', round(loso_gbm(df3, f3),2), '(harness 63.050)')

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

class GBM:
    def __init__(self, rounds=300, lr=0.1, depth=4, lam=1.0, min_leaf=20, bins=32, seed=0):
        self.p = dict(rounds=rounds, lr=lr, depth=depth, lam=lam, min_leaf=min_leaf, bins=bins, seed=seed)
    def fit(self, X, y):
        p = self.p; n, m = X.shape
        self.edges = []
        Xb = np.zeros((n, m), dtype=np.int8)
        for j in range(m):
            col = X[:, j]
            qs = np.quantile(col, np.linspace(0, 1, p['bins']+1))
            edges = np.unique(qs)
            if len(edges) < 2: edges = np.array([col.min()-1, col.max()+1])
            b = np.clip(np.searchsorted(edges[1:-1], col, side='right'), 0, len(edges)-2)
            Xb[:, j] = b; self.edges.append(edges)
        pred = np.full(n, y.mean()); self.trees = []
        for t in range(p['rounds']):
            g = pred - y
            nodes = [(np.arange(n), 0)]; tree = []
            while nodes:
                idx, d = nodes.pop(0)
                G = g[idx].sum(); H = len(idx)
                if d >= p['depth'] or H < 2*p['min_leaf'] or np.abs(g[idx]).max() < 1e-12:
                    tree.append(('L', -G/(H+p['lam']))); continue
                best = (None, -1e18, None)
                for j in range(m):
                    b = Xb[idx, j]
                    hg = np.bincount(b, weights=g[idx], minlength=p['bins'])[:p['bins']]
                    hn = np.bincount(b, minlength=p['bins'])[:p['bins']].astype(float)
                    cg = np.cumsum(hg); cn = np.cumsum(hn)
                    gl = cg[:-1]; nl = cn[:-1]; gr = G-gl; nr = H-nl
                    valid = (nl >= p['min_leaf']) & (nr >= p['min_leaf'])
                    if not valid.any(): continue
                    score = np.where(valid, gl**2/(nl+p['lam']) + gr**2/(nr+p['lam']), -1e18)
                    k = int(np.argmax(score))
                    if score[k] > best[1]:
                        thr = self.edges[j][k+1]
                        best = (j, score[k], thr)
                j, sc, thr = best
                if j is None:
                    tree.append(('L', -G/(H+p['lam']))); continue
                mask = X[idx, j] <= thr
                tree.append(('S', j, thr))
                nodes.append((idx[mask], d+1)); nodes.append((idx[~mask], d+1))
            cur = np.zeros(n, dtype=int); upd = np.zeros(n)
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    upd[cur == i] = node[1]
            pred = pred + p['lr'] * upd
            self.trees.append(tree)
        self.base = y.mean()
        return self
    def predict(self, X):
        out = np.full(len(X), self.base)
        for tree in self.trees:
            cur = np.zeros(len(X), dtype=int); leaf = np.zeros(len(X))
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    leaf[cur == i] = node[1]
            out += self.p['lr'] * leaf
        return out

def loso_gbm(df, feats, rounds=200, depth=4, lr=0.1):
    tr_days = sorted(df.snapshot_day.unique())
    X = df[feats].astype(float).values
    y = df.future_spend_4w.values; days = df.snapshot_day.values
    errs = []
    for d in tr_days:
        m = days != d
        mdl = GBM(rounds=rounds, depth=depth, lr=lr).fit(X[m], y[m])
        errs.append(float(np.mean(np.abs(mdl.predict(X[~m]) - y[~m]))))
    return float(np.mean(errs))

def tonum(df, cols):
    for c in cols:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = df[c].astype('category').cat.codes.astype(float).replace(-1.0, np.nan)
    return df

b = A.baseline_features(); tt = A.train_targets()
bfeats = [c for c in b.columns if c not in ('household_key','snapshot_day')]
db = b.merge(tt, on=['household_key','snapshot_day'])
db = tonum(db, bfeats)
print('E000 GBM LOSO:', round(loso_gbm(db, bfeats),2), '(harness 92.446)', flush=True)

t3 = A.load_saved('e003_catmix.parquet')
df3 = t3.merge(tt, on=['household_key','snapshot_day'])
f3 = [c for c in t3.columns if c not in ('household_key','snapshot_day')]
df3 = tonum(df3, f3)
print('E003 GBM LOSO:', round(loso_gbm(df3, f3),2), '(harness 63.050)')

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

class GBM:
    def __init__(self, rounds=300, lr=0.1, depth=4, lam=1.0, min_leaf=20, bins=32):
        self.p = dict(rounds=rounds, lr=lr, depth=depth, lam=lam, min_leaf=min_leaf, bins=bins)
    def fit(self, X, y):
        p = self.p; n, m = X.shape
        self.edges = []
        Xb = np.zeros((n, m), dtype=np.int8)
        for j in range(m):
            col = X[:, j]
            qs = np.quantile(col, np.linspace(0, 1, p['bins']+1))
            edges = np.unique(qs)
            if len(edges) < 2: edges = np.array([col.min()-1, col.max()+1])
            b = np.clip(np.searchsorted(edges[1:-1], col, side='right'), 0, len(edges)-2)
            Xb[:, j] = b; self.edges.append(edges)
        pred = np.full(n, y.mean()); self.trees = []
        for t in range(p['rounds']):
            g = pred - y
            nodes = [(np.arange(n), 0)]; tree = []
            while nodes:
                idx, d = nodes.pop(0)
                G = g[idx].sum(); H = len(idx)
                if d >= p['depth'] or H < 2*p['min_leaf']:
                    tree.append(('L', -G/(H+p['lam']))); continue
                best = (None, -1e18, None)
                for j in range(m):
                    b = Xb[idx, j]
                    hg = np.bincount(b, weights=g[idx], minlength=p['bins'])[:p['bins']]
                    hn = np.bincount(b, minlength=p['bins'])[:p['bins']].astype(float)
                    cg = np.cumsum(hg); cn = np.cumsum(hn)
                    gl = cg[:-1]; nl = cn[:-1]; gr = G-gl; nr = H-nl
                    valid = (nl >= p['min_leaf']) & (nr >= p['min_leaf'])
                    if not valid.any(): continue
                    score = np.where(valid, gl**2/(nl+p['lam']) + gr**2/(nr+p['lam']), -1e18)
                    k = int(np.argmax(score))
                    if score[k] > best[1]:
                        thr = self.edges[j][k+1]
                        best = (j, score[k], thr)
                j, sc, thr = best
                if j is None:
                    tree.append(('L', -G/(H+p['lam']))); continue
                mask = X[idx, j] <= thr
                tree.append(('S', j, thr))
                nodes.append((idx[mask], d+1)); nodes.append((idx[~mask], d+1))
            cur = np.zeros(n, dtype=int); upd = np.zeros(n)
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    upd[cur == i] = node[1]
            pred = pred + p['lr'] * upd
            self.trees.append(tree)
        self.base = y.mean()
        return self
    def predict(self, X):
        out = np.full(len(X), self.base)
        for tree in self.trees:
            cur = np.zeros(len(X), dtype=int); leaf = np.zeros(len(X))
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    leaf[cur == i] = node[1]
            out += self.p['lr'] * leaf
        return out

def loso_gbm(df, feats, rounds=200, depth=4, lr=0.1):
    tr_days = sorted(df.snapshot_day.unique())
    X = df[feats].astype(float).values
    y = df.future_spend_4w.values; days = df.snapshot_day.values
    errs = []
    for d in tr_days:
        m = days != d
        mdl = GBM(rounds=rounds, depth=depth, lr=lr).fit(X[m], y[m])
        errs.append(float(np.mean(np.abs(mdl.predict(X[~m]) - y[~m]))))
    return float(np.mean(errs))

def tonum(df, cols):
    for c in cols:
        dt = df[c].dtype
        if not (np.issubdtype(dt, np.number) or dt == bool):
            df[c] = pd.Categorical(df[c].fillna('__NA__')).codes.astype(float)
            df.loc[df[c] < 0, c] = np.nan
    return df

b = A.baseline_features(); tt = A.train_targets()
bfeats = [c for c in b.columns if c not in ('household_key','snapshot_day')]
db = b.merge(tt, on=['household_key','snapshot_day'])
db = tonum(db, bfeats)
print('E000 GBM LOSO:', round(loso_gbm(db, bfeats),2), '(harness 92.446)', flush=True)

t3 = A.load_saved('e003_catmix.parquet')
df3 = t3.merge(tt, on=['household_key','snapshot_day'])
f3 = [c for c in t3.columns if c not in ('household_key','snapshot_day')]
df3 = tonum(df3, f3)
print('E003 GBM LOSO:', round(loso_gbm(df3, f3),2), '(harness 63.050)')

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

class GBM:
    def __init__(self, rounds=300, lr=0.1, depth=4, lam=1.0, min_leaf=20, bins=32):
        self.p = dict(rounds=rounds, lr=lr, depth=depth, lam=lam, min_leaf=min_leaf, bins=bins)
    def fit(self, X, y):
        p = self.p; n, m = X.shape
        self.edges = []
        Xb = np.zeros((n, m), dtype=np.int8)
        for j in range(m):
            col = X[:, j]
            qs = np.quantile(col, np.linspace(0, 1, p['bins']+1))
            edges = np.unique(qs)
            if len(edges) < 2: edges = np.array([col.min()-1, col.max()+1])
            b = np.clip(np.searchsorted(edges[1:-1], col, side='right'), 0, len(edges)-2)
            Xb[:, j] = b; self.edges.append(edges)
        pred = np.full(n, y.mean()); self.trees = []
        for t in range(p['rounds']):
            g = pred - y
            nodes = [(np.arange(n), 0)]; tree = []
            while nodes:
                idx, d = nodes.pop(0)
                G = g[idx].sum(); H = len(idx)
                if d >= p['depth'] or H < 2*p['min_leaf']:
                    tree.append(('L', -G/(H+p['lam']))); continue
                best = (None, -1e18, None)
                for j in range(m):
                    b = Xb[idx, j]
                    hg = np.bincount(b, weights=g[idx], minlength=p['bins'])[:p['bins']]
                    hn = np.bincount(b, minlength=p['bins'])[:p['bins']].astype(float)
                    cg = np.cumsum(hg); cn = np.cumsum(hn)
                    gl = cg[:-1]; nl = cn[:-1]; gr = G-gl; nr = H-nl
                    valid = (nl >= p['min_leaf']) & (nr >= p['min_leaf'])
                    if not valid.any(): continue
                    score = np.where(valid, gl**2/(nl+p['lam']) + gr**2/(nr+p['lam']), -1e18)
                    k = int(np.argmax(score))
                    if score[k] > best[1]:
                        thr = self.edges[j][k+1]
                        best = (j, score[k], thr)
                j, sc, thr = best
                if j is None:
                    tree.append(('L', -G/(H+p['lam']))); continue
                mask = X[idx, j] <= thr
                tree.append(('S', j, thr))
                nodes.append((idx[mask], d+1)); nodes.append((idx[~mask], d+1))
            cur = np.zeros(n, dtype=int); upd = np.zeros(n)
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    upd[cur == i] = node[1]
            pred = pred + p['lr'] * upd
            self.trees.append(tree)
        self.base = y.mean()
        return self
    def predict(self, X):
        out = np.full(len(X), self.base)
        for tree in self.trees:
            cur = np.zeros(len(X), dtype=int); leaf = np.zeros(len(X))
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    leaf[cur == i] = node[1]
            out += self.p['lr'] * leaf
        return out

def loso_gbm(df, feats, rounds=200, depth=4, lr=0.1):
    tr_days = sorted(df.snapshot_day.unique())
    X = df[feats].astype(float).values
    y = df.future_spend_4w.values; days = df.snapshot_day.values
    errs = []
    for d in tr_days:
        m = days != d
        mdl = GBM(rounds=rounds, depth=depth, lr=lr).fit(X[m], y[m])
        errs.append(float(np.mean(np.abs(mdl.predict(X[~m]) - y[~m]))))
    return float(np.mean(errs))

def tonum(df, cols):
    for c in cols:
        if df[c].dtype.kind not in 'ifb':
            df[c] = pd.Categorical(df[c].fillna('__NA__')).codes.astype(float)
            df.loc[df[c] < 0, c] = np.nan
    return df

b = A.baseline_features(); tt = A.train_targets()
bfeats = [c for c in b.columns if c not in ('household_key','snapshot_day')]
db = b.merge(tt, on=['household_key','snapshot_day'])
db = tonum(db, bfeats)
print('E000 GBM LOSO:', round(loso_gbm(db, bfeats),2), '(harness 92.446)', flush=True)

t3 = A.load_saved('e003_catmix.parquet')
df3 = t3.merge(tt, on=['household_key','snapshot_day'])
f3 = [c for c in t3.columns if c not in ('household_key','snapshot_day')]
df3 = tonum(df3, f3)
print('E003 GBM LOSO:', round(loso_gbm(df3, f3),2), '(harness 63.050)')

# ---- cell ----
import pandas as pd, numpy as np, agent_api as A

class GBM:
    def __init__(self, rounds=300, lr=0.1, depth=4, lam=1.0, min_leaf=20, bins=32):
        self.p = dict(rounds=rounds, lr=lr, depth=depth, lam=lam, min_leaf=min_leaf, bins=bins)
    def fit(self, X, y):
        p = self.p; n, m = X.shape
        self.edges = []
        Xb = np.zeros((n, m), dtype=np.int8)
        for j in range(m):
            col = X[:, j]
            qs = np.quantile(col, np.linspace(0, 1, p['bins']+1))
            edges = np.unique(qs)
            if len(edges) < 2: edges = np.array([col.min()-1, col.max()+1])
            b = np.clip(np.searchsorted(edges[1:-1], col, side='right'), 0, len(edges)-2)
            Xb[:, j] = b; self.edges.append(edges)
        pred = np.full(n, y.mean()); self.trees = []
        for t in range(p['rounds']):
            g = pred - y
            nodes = [(np.arange(n), 0)]; tree = []
            while nodes:
                idx, d = nodes.pop(0)
                G = g[idx].sum(); H = len(idx)
                if d >= p['depth'] or H < 2*p['min_leaf']:
                    tree.append(('L', -G/(H+p['lam']))); continue
                best = (None, -1e18, None)
                for j in range(m):
                    b = Xb[idx, j]
                    hg = np.bincount(b, weights=g[idx], minlength=p['bins'])[:p['bins']]
                    hn = np.bincount(b, minlength=p['bins'])[:p['bins']].astype(float)
                    cg = np.cumsum(hg); cn = np.cumsum(hn)
                    gl = cg[:-1]; nl = cn[:-1]; gr = G-gl; nr = H-nl
                    valid = (nl >= p['min_leaf']) & (nr >= p['min_leaf'])
                    if not valid.any(): continue
                    score = np.where(valid, gl**2/(nl+p['lam']) + gr**2/(nr+p['lam']), -1e18)
                    k = int(np.argmax(score))
                    if score[k] > best[1]:
                        thr = self.edges[j][k+1]
                        best = (j, score[k], thr)
                j, sc, thr = best
                if j is None:
                    tree.append(('L', -G/(H+p['lam']))); continue
                mask = X[idx, j] <= thr
                tree.append(('S', j, thr))
                nodes.append((idx[mask], d+1)); nodes.append((idx[~mask], d+1))
            cur = np.zeros(n, dtype=int); upd = np.zeros(n)
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    upd[cur == i] = node[1]
            pred = pred + p['lr'] * upd
            self.trees.append(tree)
        self.base = y.mean()
        return self
    def predict(self, X):
        out = np.full(len(X), self.base)
        for tree in self.trees:
            cur = np.zeros(len(X), dtype=int); leaf = np.zeros(len(X))
            for i, node in enumerate(tree):
                if node[0] == 'S':
                    gl = X[:, node[1]] <= node[2]
                    msk = cur == i
                    cur[gl & msk] = 2*i+1; cur[(~gl) & msk] = 2*i+2
                else:
                    leaf[cur == i] = node[1]
            out += self.p['lr'] * leaf
        return out

def loso_gbm(df, feats, rounds=200, depth=4, lr=0.1):
    tr_days = sorted(df.snapshot_day.unique())
    X = df[feats].astype(float).values
    y = df.future_spend_4w.values; days = df.snapshot_day.values
    errs = []
    for d in tr_days:
        m = days != d
        mdl = GBM(rounds=rounds, depth=depth, lr=lr).fit(X[m], y[m])
        errs.append(float(np.mean(np.abs(mdl.predict(X[~m]) - y[~m]))))
    return float(np.mean(errs))

def tonum(df, cols):
    for c in cols:
        if df[c].dtype.kind not in 'ifb':
            codes = pd.Categorical(df[c].astype(str).fillna('__NA__')).codes.astype(float)
            codes[codes < 0] = np.nan
            df[c] = codes
    return df

b = A.baseline_features(); tt = A.train_targets()
bfeats = [c for c in b.columns if c not in ('household_key','snapshot_day')]
db = b.merge(tt, on=['household_key','snapshot_day'])
db = tonum(db, bfeats)
print('E000 GBM LOSO:', round(loso_gbm(db, bfeats),2), '(harness 92.446)', flush=True)

t3 = A.load_saved('e003_catmix.parquet')
df3 = t3.merge(tt, on=['household_key','snapshot_day'])
f3 = [c for c in t3.columns if c not in ('household_key','snapshot_day')]
df3 = tonum(df3, f3)
print('E003 GBM LOSO:', round(loso_gbm(df3, f3),2), '(harness 63.050)')