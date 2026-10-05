
import agent_api, pandas as pd, numpy as np
e13 = agent_api.load_saved('e013_stationary.parquet')
e15 = agent_api.load_saved('e015_norm_windows.parquet')
e17 = agent_api.load_saved('e017_reversion.parquet')
print('shapes', e13.shape, e15.shape, e17.shape)
print('cols e13:', sorted(e13.columns))
print('dropped by E015 (in e13 not e15):', sorted(set(e13.columns)-set(e15.columns)))
print('e17 adds:', sorted(set(e17.columns)-set(e13.columns)))
tt = agent_api.train_targets()
y = tt.future_spend_4w
print('n=%d zero=%.3f mean=%.1f med=%.1f p90=%.0f max=%.0f' % (len(tt),(y==0).mean(),y.mean(),y.median(),y.quantile(.9),y.max()))
print(tt.groupby('snapshot_day').future_spend_4w.agg(['size','mean','median']).round(1))
print('e13 rows per snapshot:'); print(e13.snapshot_day.value_counts().sort_index().to_dict())
m = e13.merge(tt, on=['household_key','snapshot_day'])
num = m.drop(columns=['household_key','snapshot_day','future_spend_4w']).select_dtypes('number')
pc = num.corrwith(m.future_spend_4w)
print('pearson with target, bottom15:'); print(pc.sort_values().head(15).round(3).to_string())
print('top15:'); print(pc.sort_values().tail(15).round(3).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e13 = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = e13.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>431]
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values

feats = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
Xtr = tr[feats].astype(float).copy(); Xva = va[feats].astype(float).copy()
Xtr = Xtr.fillna(Xtr.median()); Xva = Xva.fillna(Xtr.median())
mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
Xtr_s = ((Xtr-mu)/sd).values; Xva_s = ((Xva-mu)/sd).values
Xtr_s = np.c_[np.ones(len(Xtr_s)), Xtr_s]; Xva_s = np.c_[np.ones(len(Xva_s)), Xva_s]

def ridge_fit(X,y,lam):
    A = X.T@X + lam*np.eye(X.shape[1]); A[0,0]-=lam
    return np.linalg.solve(A, X.T@y)

print('baseline mean-pred val MAE: %.2f' % np.abs(yva-ytr.mean()).mean())
for lam in [0.3,1,3,10,30,100,300,1000]:
    b = ridge_fit(Xtr_s,ytr,lam)
    print('lam %6.1f  val MAE %.3f  train MAE %.3f' % (lam, np.abs(Xva_s@b-yva).mean(), np.abs(Xtr_s@b-ytr).mean()))

b = ridge_fit(Xtr_s,ytr,30)
res = ytr - Xtr_s@b
print('\nval MAE lam30: %.3f' % np.abs(Xva_s@b-yva).mean())
print('bias by snapshot (val):'); 
for d in sorted(va.snapshot_day.unique()):
    k = va.snapshot_day==d
    print(' day %d: mean y %.1f mean pred %.1f' % (d, yva[k].mean(), (Xva_s@b)[k].mean()))
# residual correlation with features (in-sample, indicative only)
rc = pd.Series({f: np.corrcoef(Xtr[f], res)[0,1] for f in feats}).sort_values()
print('\nresid corr bottom10:'); print(rc.head(10).round(3).to_string())
print('resid corr top10:'); print(rc.tail(10).round(3).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e13 = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = e13.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>431]
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values
feats = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
def prep(df):
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    return X.astype(float)
Xtr = prep(tr); Xva = prep(va)
med = Xtr.median(); Xtr = Xtr.fillna(med); Xva = Xva.fillna(med)
mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
Xtr_s = ((Xtr-mu)/sd).values; Xva_s = ((Xva-mu)/sd).values
Xtr_s = np.c_[np.ones(len(Xtr_s)), Xtr_s]; Xva_s = np.c_[np.ones(len(Xva_s)), Xva_s]
def ridge_fit(X,y,lam):
    A = X.T@X + lam*np.eye(X.shape[1]); A[0,0]-=lam
    return np.linalg.solve(A, X.T@y)
for lam in [0.3,1,3,10,30,100,300,1000]:
    b = ridge_fit(Xtr_s,ytr,lam)
    print('lam %6.1f  val MAE %.3f' % (lam, np.abs(Xva_s@b-yva).mean()))
b = ridge_fit(Xtr_s,ytr,30)
pred = Xva_s@b
print('\nval MAE lam30: %.3f' % np.abs(pred-yva).mean())
for d in sorted(va.snapshot_day.unique()):
    k = (va.snapshot_day==d).values
    print(' day %d: mean y %.1f mean pred %.1f' % (d, yva[k].mean(), pred[k].mean()))
res = ytr - Xtr_s@b
rc = pd.Series({f: np.corrcoef(Xtr[f], res)[0,1] for f in feats}).sort_values()
print('\nresid corr bottom10:'); print(rc.head(10).round(3).to_string())
print('resid corr top10:'); print(rc.tail(10).round(3).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e13 = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = e13.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>431]
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values
feats = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
def prep(df):
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    return X.astype(float)
Xtr = prep(tr); Xva = prep(va)
print('NaN cols in train:', Xtr.isna().sum()[Xtr.isna().sum()>0].to_dict())
print('NaN cols in val:', Xva.isna().sum()[Xva.isna().sum()>0].to_dict())
print('inf check:', np.isinf(Xtr.values).sum(), np.isinf(Xva.values).sum())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e13 = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = e13.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>431]
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values
feats = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
def prep(df):
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    return X.astype(float)
Xtr = prep(tr); Xva = prep(va)
med = Xtr.median(); Xtr = Xtr.fillna(med); Xva = Xva.fillna(med)
print('any NaN after fill:', Xtr.isna().sum().sum(), Xva.isna().sum().sum())
mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
Xtr_s = ((Xtr-mu)/sd).values; Xva_s = ((Xva-mu)/sd).values
print('nan in Xtr_s:', np.isnan(Xtr_s).sum(), 'nan Xva_s:', np.isnan(Xva_s).sum())
lam=30
X1 = np.c_[np.ones(len(Xtr_s)), Xtr_s]
A = X1.T@X1 + lam*np.eye(X1.shape[1]); A[0,0]-=lam
print('A nan:', np.isnan(A).sum())
b = np.linalg.solve(A, X1.T@ytr)
print('b nan:', np.isnan(b).sum())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e13 = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = e13.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>431]
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values
feats = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
def prep(df):
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    return X.astype(float)
Xtr = prep(tr).fillna(prep(tr).median()); Xva = prep(va)
med = prep(tr).median(); Xtr = prep(tr).fillna(med); Xva = prep(va).fillna(med)
mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
Xtr_s = ((Xtr-mu)/sd).values; Xva_s = ((Xva-mu)/sd).values
X1 = np.c_[np.ones(len(Xtr_s)), Xtr_s]; Xv1 = np.c_[np.ones(len(Xva_s)), Xva_s]
def ridge_fit(X,y,lam):
    A = X.T@X + lam*np.eye(X.shape[1]); A[0,0]-=lam
    return np.linalg.solve(A, X.T@y)
for lam in [0.3,1,3,10,30,100,300,1000]:
    b = ridge_fit(X1,ytr,lam)
    print('lam %6.1f  val MAE %.3f' % (lam, np.abs(Xv1@b-yva).mean()))
b = ridge_fit(X1,ytr,30)
pred = Xv1@b
print('\nlam30 val MAE: %.3f' % np.abs(pred-yva).mean())
for d in sorted(va.snapshot_day.unique()):
    k = (va.snapshot_day==d).values
    print(' day %d: mean y %7.1f  mean pred %7.1f  MAE %.1f' % (d, yva[k].mean(), pred[k].mean(), np.abs(pred[k]-yva[k]).mean()))
res = ytr - X1@b
rc = pd.Series({f: np.corrcoef(Xtr[f], res)[0,1] for f in feats}).sort_values()
print('\nresid corr bottom10:'); print(rc.head(10).round(3).to_string())
print('resid corr top10:'); print(rc.tail(10).round(3).to_string())
# MAE by predicted decile on val
q = pd.qcut(pred, 10, duplicates='drop')
print('\nval MAE by pred decile:')
print(pd.DataFrame({'pred':pred,'y':yva,'q':q}).groupby('q').apply(lambda g: pd.Series({'n':len(g),'mean_pred':g.pred.mean(),'mean_y':g.y.mean(),'mae':np.abs(g.pred-g.y).mean()})).round(1).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def prep_table(path):
    df = agent_api.load_saved(path)
    feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    X = X.astype(float)
    return df, X

def eval_proxy(df, X, lam=30, val_days=(403,431)):
    tt = agent_api.train_targets()
    m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
    y = m.future_spend_4w.values
    trm = m.snapshot_day <= 375; vam = m.snapshot_day.isin(val_days)
    Xtr, Xva = X[trm.values], X[vam.values]
    med = Xtr.median(); Xtr = Xtr.fillna(med); Xva = Xva.fillna(med)
    mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
    A = np.c_[np.ones(trm.sum()), ((Xtr-mu)/sd).values]
    B = np.c_[np.ones(vam.sum()), ((Xva-mu)/sd).values]
    M = A.T@A + lam*np.eye(A.shape[1]); M[0,0]-=lam
    b = np.linalg.solve(M, A.T@y[trm.values])
    return np.abs(B@b - y[vam.values]).mean()

tables = {'e013':'e013_stationary.parquet','e011':'e011_pruned_basket.parquet',
          'e015':'e015_norm_windows.parquet','e017':'e017_reversion.parquet',
          'e008':'e008_log_transform.parquet','e010':'e010_store_mix.parquet'}
harness = {'e013':60.788,'e011':60.895,'e015':60.944,'e017':60.808,'e008':60.953,'e010':60.975}
for lam in [10,30,100]:
    scores = {k: eval_proxy(*prep_table(p), lam=lam) for k,p in tables.items()}
    rank = sorted(scores, key=scores.get)
    print('lam', lam, {k:round(v,2) for k,v in scores.items()}, 'proxy rank:', rank)
print('harness rank:', sorted(harness, key=harness.get))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
X = df[feats].copy()
for c in X.columns:
    if X[c].dtype == object:
        X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
X = X.astype(float)
trm = (m.snapshot_day<=375).values; vam = m.snapshot_day.isin([403,431]).values
Xtr, Xva = X[trm], X[vam]
med = Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
A = np.c_[np.ones(trm.sum()), ((Xtr-mu)/sd).values]; B = np.c_[np.ones(vam.sum()), ((Xva-mu)/sd).values]
lam = 30
M = A.T@A + lam*np.eye(A.shape[1]); M[0,0]-=lam
b = np.linalg.solve(M, A.T@y[trm])
pred = B@b; yva = y[vam]
print('pseudo-val MAE %.2f' % np.abs(pred-yva).mean())
dv = m.snapshot_day[vam]
for d in sorted(dv.unique()):
    k = (dv==d).values
    print(' day %d: n %d  mean y %7.1f mean pred %7.1f  MAE %.1f' % (d,k.sum(),yva[k].mean(),pred[k].mean(),np.abs(pred[k]-yva[k]).mean()))
z = yva==0
print('\nzero rows: n=%d (%.1f%%)  MAE %.1f  mean pred %.1f' % (z.sum(),100*z.mean(),np.abs(pred[z]-yva[z]).mean(),pred[z].mean()))
print('nonzero rows: MAE %.1f' % np.abs(pred[~z]-yva[~z]).mean())
print('share of total MAE from zero rows: %.2f' % (np.abs(pred[z]-yva[z]).sum()/np.abs(pred-yva).sum()))
res = y[trm] - A@b
Xtr_df = pd.DataFrame(Xtr, columns=feats)
rc = Xtr_df.corrwith(pd.Series(res)).sort_values()
print('\nresid corr bottom12:'); print(rc.head(12).round(3).to_string())
print('resid corr top12:'); print(rc.tail(12).round(3).to_string())
q = pd.qcut(yva, 10, duplicates='drop')
g = pd.DataFrame({'pred':pred,'y':yva,'q':q}).groupby('q',observed=True).apply(lambda t: pd.Series({'n':len(t),'mean_y':t.y.mean(),'mean_pred':t.pred.mean(),'mae':np.abs(t.pred-t.y).mean()}))
print('\nMAE by true-spend decile:'); print(g.round(1).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
X = df[feats].copy()
for c in X.columns:
    if X[c].dtype == object:
        X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
X = X.astype(float)
trm = (m.snapshot_day<=375).values; vam = m.snapshot_day.isin([403,431]).values
med = X[trm].median(); Xf = X.fillna(med)

def proxy(Xnew, lam=30):
    mu, sd = Xnew[trm].mean(), Xnew[trm].std().replace(0,1)
    A = np.c_[np.ones(trm.sum()), ((Xnew[trm]-mu)/sd).values]
    B = np.c_[np.ones(vam.sum()), ((Xnew[vam]-mu)/sd).values]
    M = A.T@A + lam*np.eye(A.shape[1]); M[0,0]-=lam
    b = np.linalg.solve(M, A.T@y[trm])
    return np.abs(B@b - y[vam]).mean()

base = proxy(Xf); print('base E013 proxy: %.3f' % base)

def hinge(s, t): return np.maximum(s - t, 0)
def add_hinges(Xnew, spec):
    for col, thrs in spec.items():
        for t in thrs:
            Xnew[col+'_h%d'%t] = np.maximum(Xnew[col] - t, 0)
    return Xnew

specA = {'dec_28':[25,75,150,300,600], 'dec_56':[25,75,150,300,600], 'spend_84':[50,150,400,800],
         'z_rate84':[25,75,200], 'weekly_mean_12':[25,75,200]}
print('A hinges tail: %.3f' % proxy(add_hinges(Xf.copy(), specA)))
specB = {'dec_28':[10,25,50,100,200,400], 'spend_84':[25,75,200,500,1000]}
print('B hinges dec28+sp84: %.3f' % proxy(add_hinges(Xf.copy(), specB)))
# winsorize heavy spend cols at train p99
Xw = Xf.copy()
sp_cols = [c for c in Xf.columns if c.startswith(('spend','dec_','weekly','avg_'))]
for c in sp_cols:
    Xw[c] = Xw[c].clip(upper=Xw[c][trm].quantile(0.99))
print('C winsorize: %.3f' % proxy(Xw))
print('D winsor+hingesB: %.3f' % proxy(add_hinges(Xw.copy(), specB)))
# log versions of spend cols
Xl = Xf.copy()
for c in sp_cols:
    Xl[c] = np.log1p(Xl[c])
print('E log spend cols: %.3f' % proxy(Xl))
print('F log+hingesB: %.3f' % proxy(add_hinges(Xl.copy(), specB)))


# ---- cell ----

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


# ---- cell ----

import agent_api, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
tt = agent_api.train_targets()

def prep(df):
    feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    return X.astype(float).values

def gbm_proxy(df, depth=3, lr=0.1, n_trees=150, l2=1.0, min_leaf=20, seed=0):
    X = prep(df)
    m = df[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')
    y = m.future_spend_4w.values
    trm = (m.snapshot_day<=375).values; vam = m.snapshot_day.isin([403,431]).values
    Xtr, Xva = X[trm], X[vam]
    med = np.nanmedian(Xtr, 0); Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
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
        feats = rng.choice(F, max(2,int(F*0.7)), replace=False)
        featL=[]; thrL=[]; leftL=[]; rightL=[]; leafL=[]
        def new_leaf(idx):
            leafL.append(-g[idx].sum()/(len(idx)+l2)); featL.append(-1); return len(leafL)-1
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
        featA = np.array(featL); thrA = np.array(thrL)
        leftA = np.array(leftL); rightA = np.array(rightL); leafA = np.array(leafL)
        trees.append((featA,thrA,leftA,rightA,leafA))
        cur = np.zeros(len(ytr), dtype=int)
        while True:
            mask = featA[cur] >= 0
            if not mask.any(): break
            rows = np.where(mask)[0]; f = featA[cur[rows]]; b = Btr[rows, f]
            cur[rows] = np.where(b<=thrA[cur[rows]], leftA[cur[rows]], rightA[cur[rows]])
        pred += lr*leafA[cur]
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

t0=time.time()
harness = {'e013':60.788,'e011':60.895,'e015':60.944,'e017':60.808,'e014':61.912,'e016':65.334}
paths = {'e013':'e013_stationary.parquet','e011':'e011_pruned_basket.parquet','e015':'e015_norm_windows.parquet',
         'e017':'e017_reversion.parquet','e014':'e014_seasonal.parquet','e016':'e016_display.parquet'}
for name in ['e013','e011','e015','e017','e014','e016']:
    print('%s harness %.2f | gbm %.2f' % (name, harness[name], gbm_proxy(agent_api.load_saved(paths[name]))))
print('%.0fs' % (time.time()-t0))


# ---- cell ----

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
