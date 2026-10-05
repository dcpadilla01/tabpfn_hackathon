import pandas as pd, numpy as np, agent_api
feats = agent_api.load_saved('feats_v4.parquet')
print('feats', feats.shape)
print(list(feats.columns))
tt = agent_api.train_targets()
print('tt', tt.shape)
print(tt.future_spend_4w.describe())
print('zero frac train targets:', round((tt.future_spend_4w==0).mean(),4))
val = feats[feats.snapshot_day>=459]
print('val rows', len(val), 'hh', val.household_key.nunique())
tset = set(tt.household_key)
for s in sorted(val.snapshot_day.unique()):
    v = val[val.snapshot_day==s]
    print('day', s, 'rows', len(v), 'cov', round(v.household_key.isin(tset).mean(),3))
m = tt.groupby('household_key').future_spend_4w.agg(['mean','count'])
f2 = feats.merge(m, left_on='household_key', right_index=True, how='left')
cols = [c for c in feats.columns if feats[c].dtype.kind in 'fi']
cort = f2[cols+['mean']].corr()['mean'].drop('mean').sort_values(key=abs, ascending=False)
print('top corr with hh mean target:')
print(cort.head(15))
p = agent_api.load_saved('pred_e011.parquet')
print('pred_e011', p.shape, list(p.columns))
print(p.head(3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
print('dtypes non-numeric:', [c for c in feats.columns if feats[c].dtype.kind not in 'fi'])
# encode any object cols
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feats2 = df.drop(columns=['future_spend_4w'])
y = df['future_spend_4w'].values
feat_cols = [c for c in feats2.columns if c not in ('household_key','snapshot_day')]
X = feats2[feat_cols].astype(float).values

def fit_q(Xtr, ytr, Xte, alpha=0.5, depth=5, mcw=40, lr=0.08, n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
                         max_depth=depth, min_child_weight=mcw, learning_rate=lr,
                         n_estimators=n, subsample=0.9, colsample_bytree=0.8,
                         n_jobs=8, tree_method='hist')
    m.fit(Xtr, ytr); return m.predict(Xte)

def fit_c(Xtr, ytr, Xte):
    m = xgb.XGBClassifier(objective='binary:logistic', max_depth=4, min_child_weight=40,
                          learning_rate=0.08, n_estimators=300, subsample=0.9,
                          colsample_bytree=0.8, n_jobs=8, tree_method='hist')
    m.fit(Xtr, ytr); return m.predict_proba(Xte)[:,1]

# internal validation: train on snapshots <=403, test on 431
tr = df.snapshot_day <= 403
te = df.snapshot_day == 431
itr, ite = np.where(tr)[0], np.where(te)[0]
Xtr, ytr, Xte, yte = X[itr], y[itr], X[ite], y[ite]
print('train rows', len(itr), 'test rows', len(ite), 'test MAE of exp4w_blend:', round(np.abs(feats2.exp4w_blend.values[ite]-yte).mean(),3))

# A: plain quantile model (E011 style)
pa = fit_q(Xtr, ytr, Xte)
print('A quantile all:', round(np.abs(pa-yte).mean(),3))
# B: hurdle p*q
pc = fit_c(Xtr, (ytr>0).astype(int), Xte)
pos = ytr>0
pq = fit_q(Xtr[pos], ytr[pos], Xte)
pb = pc*pq
print('B hurdle p*q:', round(np.abs(pb-yte).mean(),3))
# C: blend model with exp4w_blend
eb = feats2.exp4w_blend.values[ite]
for w in (0.2,0.3,0.4):
    print(f'C blend w={w}:', round(np.abs((1-w)*pa+w*eb-yte).mean(),3))
# D: hurdle + blend
for w in (0.2,0.3):
    print(f'D hurdle+blend w={w}:', round(np.abs((1-w)*pb+w*eb-yte).mean(),3))
# E: p-shifted quantile (0.55)
pq55 = fit_q(Xtr, ytr, Xte, alpha=0.55)
print('E q0.55 all:', round(np.abs(pq55-yte).mean(),3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
# drift check
print('mean target by snapshot_day:')
print(df.groupby('snapshot_day').future_spend_4w.agg(['mean','median']).round(1))

def fit_q(Xtr, ytr, Xte, alpha=0.5, seed=7, n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
                         max_depth=5, min_child_weight=40, learning_rate=0.08,
                         n_estimators=n, subsample=0.9, colsample_bytree=0.8,
                         n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr, ytr); return m.predict(Xte)

tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
def mae(p): return round(np.abs(p-yte).mean(),3)

# A: baseline single q0.5
pa = fit_q(Xtr, ytr, Xte); print('A q0.5:', mae(pa))
# B: log-target median
pl = np.expm1(fit_q(Xtr, np.log1p(ytr), Xte)); print('B log-median:', mae(pl))
# C: avg of log-median and raw-median
print('C avg(raw,log):', mae((pa+pl)/2))
# D: seed bagging (5 seeds raw q0.5)
pb = np.mean([fit_q(Xtr, ytr, Xte, seed=s) for s in (1,2,3,4,5)], axis=0)
print('D seed-bag q0.5:', mae(pb))
# E: multi-quantile avg 0.45/0.5/0.55
pq = np.mean([fit_q(Xtr, ytr, Xte, alpha=a) for a in (0.45,0.5,0.55)], axis=0)
print('E multi-q avg:', mae(pq))
# F: recency-weighted training (weight 2^(day-95)/336)
w = 2.0**((df.snapshot_day.values[tr]-95)/336.0)
pw = fit_q(Xtr, ytr, Xte)
m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=5,
                     min_child_weight=40, learning_rate=0.08, n_estimators=400,
                     subsample=0.9, colsample_bytree=0.8, n_jobs=8, tree_method='hist')
m.fit(Xtr, ytr, sample_weight=w); print('F recency-wt:', mae(m.predict(Xte)))
# G: bag of (raw q0.5 + log) mixes
print('G avg of D and B:', mae((pb+pl)/2))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
def mae(p): return round(np.abs(p-yte).mean(),3)
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=5,
        min_child_weight=40, learning_rate=0.08, n_estimators=400, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)

# H: HistGBR quantile
hg = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=300,
    learning_rate=0.08, max_depth=None, max_leaf_nodes=31, min_samples_leaf=60,
    l2_regularization=1.0, random_state=7).fit(Xtr, ytr)
ph = hg.predict(Xte); print('H HistGBR q0.5:', mae(ph))
# I: avg xgb(seeds) + HistGBR
pb = np.mean([xq(Xtr,ytr,Xte,seed=s) for s in (1,2,3,4,5)], axis=0)
print('I avg(xgb-bag, histgbr):', mae((pb+ph)/2), '| w 0.7/0.3:', mae(0.7*pb+0.3*ph))
# J: train only on later snapshots (day>=235)
late = np.where(df.snapshot_day<=403)[0][df.snapshot_day.values[tr]>=235]
pl2 = xq(X[late], y[late], Xte); print('J late-only:', mae(pl2))
# K: drop demographic cols
demo_idx = [i for i,c in enumerate(feat_cols) if c.startswith(('classification','homeowner','kid','has_demo'))]
Xk = np.delete(X, demo_idx, axis=1)
pk = xq(Xk[tr], ytr, Xk[te]); print('K no-demo:', mae(pk))
# L: drop snap cols
snap_idx = [i for i,c in enumerate(feat_cols) if c.startswith('snap')]
Xl = np.delete(X, snap_idx, axis=1)
pl3 = xq(Xl[tr], ytr, Xl[te]); print('L no-snap:', mae(pl3))
# M: bag with raw+log mix + histgbr
plog = np.expm1(xq(Xtr, np.log1p(ytr), Xte))
print('M best-mix avg(pb,pl,ph):', mae((pb+pl+ph)/3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
def mae(p): return round(np.abs(p-yte).mean(),3)
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=5,
        min_child_weight=40, learning_rate=0.08, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
pb = np.mean([xq(Xtr,ytr,Xte,seed=s) for s in (1,2,3,4,5)], axis=0)
print('bag:', mae(pb))
# affine calibration on internal test
from scipy.optimize import minimize
def obj(th): return np.abs(th[0]*pb+th[1]-yte).mean()
r = minimize(obj, [1.0,0.0], method='Nelder-Mead')
print('affine calib a,b:', r.x.round(4), 'mae:', round(r.fun,3))
# median bias
print('pred median', np.median(pb), 'target median', np.median(yte), 'pred mean', pb.mean(), 'tgt mean', yte.mean())
# MAE-optimal per-decile shift: bin by prediction, find best additive shift per bin
qs = np.quantile(pb, np.linspace(0,1,11))
bins = np.clip(np.searchsorted(qs, pb, side='right')-1, 0, 9)
p2 = pb.copy()
for b in range(10):
    m = bins==b
    if m.sum()>50:
        cand = np.median(yte[m])-np.median(pb[m])
        p2[m] = pb[m]+cand
print('per-decile median-shift:', mae(p2))
# seasonal feature test: spend in 28d window ending at s-364 and s-336
tr_ = agent_api.snapshot(459).table('transactions')
def seasonal(snap_day, hh_set, offsets=(336,364,392)):
    out = {}
    t = tr_[tr_.day <= snap_day]
    g = t.groupby(['household_key','day']).sales_value.sum().reset_index()
    for off in offsets:
        lo, hi = snap_day-off-27, snap_day-off
        m = (g.day>=lo)&(g.day<=hi)
        s = g[m].groupby('household_key').sales_value.sum()
        out[f'seas_{off}'] = s
    return out
# build for snapshot 431 internal test rows
hh_te = df.household_key.values[te]; sd = 431
seas = seasonal(sd, None)
Xseas = np.zeros((len(te),3))
for i,off in enumerate((336,364,392)):
    Xseas[:,i] = hh_te.map(seas[f'seas_{off}']).fillna(0).values if hasattr(hh_te,'map') else pd.Series(hh_te).map(seas[f'seas_{off}']).fillna(0).values
for i,off in enumerate((336,364,392)):
    print(f'corr seas_{off} vs yte:', round(np.corrcoef(Xseas[:,i], yte)[0,1],3), 'MAE alone:', mae(Xseas[:,i]))
Xs2 = np.hstack([Xte, Xseas])
Xs2tr = np.hstack([Xtr, np.zeros((len(tr),3))])
p_s = xq(Xs2tr, ytr, Xs2); print('with seasonal feats (zeros in train):', mae(p_s))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=0.08, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
def mae(p,yte): return round(np.abs(p-yte).mean(),3)

# diverse bag on internal split (train<=403, test 431)
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
pdiv = np.mean([xq(Xtr,ytr,Xte,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
print('diverse bag:', mae(pdiv,yte))
def calib(p, yte):
    r = minimize(lambda th: np.abs(th[0]*p+th[1]-yte).mean(), [1.0,0.0], method='Nelder-Mead')
    return r.x, r.fun
th, f = calib(pdiv, yte)
print('calib on 431:', th.round(4), 'calibrated MAE:', round(f,3))
# calibration stability: fit on other internal splits, apply to 431 preds
for cut, tst in ((375,403),(347,375),(319,347)):
    tri = np.where(df.snapshot_day <= cut)[0]; tei = np.where(df.snapshot_day == tst)[0]
    pi = np.mean([xq(X[tri],y[tri],X[tei],seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    thi, _ = calib(pi, y[tei])
    print(f'calib fit on {tst}:', thi.round(4), '-> applied to 431 preds MAE:', mae(thi[0]*pdiv+thi[1], yte))
# seed bag + calib for comparison
ps = np.mean([xq(Xtr,ytr,Xte,seed=s) for s in (1,2,3,4,5)], axis=0)
ths, fs = calib(ps, yte)
print('seed bag calib:', ths.round(4), round(fs,3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
eb = df.exp4w_blend.values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=0.08, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
ebtr, ebte = eb[tr], eb[te]
def mae(p): return round(np.abs(p-yte).mean(),3)
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
def bag(Xa,ya,Xb,**kw): return np.mean([xq(Xa,ya,Xb,seed=s,depth=d,mcw=m,**kw) for s,d,m in cfgs], axis=0)
pdiv = bag(Xtr,ytr,Xte); print('diverse bag base:', mae(pdiv))
# residual learning
pr = bag(Xtr, ytr-ebtr, Xte)
for w in (0.3,0.5,0.7,1.0):
    print(f'residual w={w}:', mae(ebte + w*pr))
# lag4-6 features: need spend in windows s-167..s-140 etc. Compute from transactions via history? Use view at snapshot. 
# Approximate: build lag4..lag6 with build_features is expensive; instead test activity-split models
act = df.spend_84.values
med = np.median(act[tr])
m_hi, m_lo = act[te] > med, act[te] <= med
p_hi = bag(Xtr[act[tr]>med], ytr[act[tr]>med], Xte[m_hi])
p_lo = bag(Xtr[act[tr]<=med], ytr[act[tr]<=med], Xte[m_lo])
psp = np.zeros(len(te)); psp[m_hi] = p_hi; psp[m_lo] = p_lo
print('activity-split:', mae(psp))
# quantile-dispersed mix: 0.5 weight on 0.4/0.6 quantiles (MAE-optimal can differ)
pq = 0.5*xq(Xtr,ytr,Xte,alpha=0.5) + 0.25*xq(Xtr,ytr,Xte,alpha=0.4) + 0.25*xq(Xtr,ytr,Xte,alpha=0.6)
print('q mix .4/.5/.6:', mae(pq))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
eb = df.exp4w_blend.values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400,lr=0.08):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=lr, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
ebtr, ebte = eb[tr], eb[te]
def mae(p): return round(np.abs(p-yte).mean(),3)
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
def bag(Xa,ya,Xb,resid=False,**kw):
    yy = ya-eb[tr] if False else ya
    return np.mean([xq(Xa,yy,Xb,seed=s,depth=d,mcw=m,**kw) for s,d,m in cfgs], axis=0)
# residual diverse bag
pr = np.mean([xq(Xtr,ytr-ebtr,Xte,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
pres = ebte+pr; print('residual diverse bag:', mae(pres))
# clipping
hi = np.quantile(ytr, 0.995)
print('clip at', hi, '->', mae(np.clip(pres,0,hi)))
# mcw 100/150 diverse
pb2 = np.mean([xq(Xtr,ytr-ebtr,Xte,seed=s,depth=d,mcw=m2) for s,d,m2 in [(7,5,100),(2,5,150),(3,4,100),(4,6,120)]], axis=0)
print('residual mcw100+:', mae(ebte+pb2))
# n=600 lr .05
pb3 = np.mean([xq(Xtr,ytr-ebtr,Xte,seed=s,depth=d,mcw=m,n=600,lr=0.05) for s,d,m in cfgs], axis=0)
print('residual n600 lr.05:', mae(ebte+pb3))
# combine residual bag + plain bag
pdiv = np.mean([xq(Xtr,ytr,Xte,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
print('avg(res,plain):', mae((pres+pdiv)/2))
# check extreme predictions
print('pred max', pres.max(), 'y max', yte.max(), 'n>1000:', (pres>1000).sum())


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
eb = df.exp4w_blend.values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400,lr=0.08):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=lr, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
def run(cut, tst):
    tri = np.where(df.snapshot_day <= cut)[0]; tei = np.where(df.snapshot_day == tst)[0]
    Xa,ya,Xb,yb,eba,ebb = X[tri],y[tri],X[tei],y[tei],eb[tri],eb[tei]
    def mae(p): return round(np.abs(p-yb).mean(),3)
    pres = eba and None
    pr = np.mean([xq(Xa,ya-eba,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    pl = np.mean([xq(Xa,ya,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    plog = np.expm1(np.mean([xq(Xa,np.log1p(ya)-np.log1p(eba),Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0))
    print(f'[{tst}] plain:{mae(pl)} res:{mae(ebb+pr)} avg(res,plain):{mae((pl+ebb+pr)/2)} 3way+logres:{mae((pl+ebb+pr+plog)/4)}')
    return mae(pl), mae(ebb+pr), mae((pl+ebb+pr)/2), mae((pl+ebb+pr+plog)/4)
r431 = run(403, 431)
r403 = run(375, 403)
r375 = run(347, 375)
print('mean across 3 splits:', np.round(np.mean([r431,r403,r375],axis=0),3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
eb = df.exp4w_blend.values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400,lr=0.08):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=lr, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
def run(cut, tst):
    tri = np.where(df.snapshot_day <= cut)[0]; tei = np.where(df.snapshot_day == tst)[0]
    Xa,ya,Xb,yb,eba,ebb = X[tri],y[tri],X[tei],y[tei],eb[tri],eb[tei]
    def mae(p): return round(np.abs(p-yb).mean(),3)
    pr = np.mean([xq(Xa,ya-eba,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    pl = np.mean([xq(Xa,ya,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    plog = np.expm1(np.mean([xq(Xa,np.log1p(ya)-np.log1p(eba),Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0))
    a1,a2,a3,a4 = mae(pl), mae(ebb+pr), mae((pl+ebb+pr)/2), mae((pl+ebb+pr+plog)/4)
    print(f'[{tst}] plain:{a1} res:{a2} avg(res,plain):{a3} 3way+logres:{a4}')
    return (a1,a2,a3,a4)
r431 = run(403, 431)
r403 = run(375, 403)
r375 = run(347, 375)
print('mean across 3 splits:', np.round(np.mean([r431,r403,r375],axis=0),3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
eb = df.exp4w_blend.values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400,lr=0.08):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=lr, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
res_out = []
for cut, tst in ((403,431),(375,403)):
    tri = np.where(df.snapshot_day <= cut)[0]; tei = np.where(df.snapshot_day == tst)[0]
    Xa,ya,Xb,yb,eba,ebb = X[tri],y[tri],X[tei],y[tei],eb[tri],eb[tei]
    def mae(p): return round(np.abs(p-yb).mean(),3)
    pr = np.mean([xq(Xa,ya-eba,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    pl = np.mean([xq(Xa,ya,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
    pres = ebb+pr
    ws = []
    for w in (0.3,0.4,0.5,0.6,0.7):
        ws.append(mae(w*pres+(1-w)*pl))
    print(f'[{tst}] w sweep 0.3..0.7:', ws)
    res_out.append(ws)
print('mean w sweep:', np.round(np.mean(res_out,axis=0),3))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
eb = df.exp4w_blend.values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400,lr=0.08):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=lr, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
tri = np.where(df.snapshot_day <= 431)[0]
tei = np.where(df.snapshot_day >= 459)[0]
Xa,ya,Xb,eba = X[tri],y[tri],X[tei],eb[tei]
pr = np.mean([xq(Xa,ya-eba,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
pl = np.mean([xq(Xa,ya,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
pred = 0.4*(eba+pr) + 0.6*pl
out = df.iloc[tei][['household_key','snapshot_day']].copy()
out['prediction'] = pred
path = agent_api.save_table(out, 'pred_e013.parquet')
print('saved', path, out.shape, 'pred range', pred.min().round(1), pred.max().round(1))


# ---- cell ----
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
eb = df.exp4w_blend.values
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7,depth=5,mcw=40,n=400,lr=0.08):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
        min_child_weight=mcw, learning_rate=lr, n_estimators=n, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)
cfgs = [(7,5,40),(2,4,20),(3,6,60),(4,5,80),(5,4,40),(8,6,40),(9,5,20),(10,4,80)]
tri = np.where(df.snapshot_day <= 431)[0]
tei = np.where(df.snapshot_day >= 459)[0]
Xa,ya,Xb,ebtr,ebte = X[tri],y[tri],X[tei],eb[tri],eb[tei]
pr = np.mean([xq(Xa,ya-ebtr,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
pl = np.mean([xq(Xa,ya,Xb,seed=s,depth=d,mcw=m) for s,d,m in cfgs], axis=0)
pred = 0.4*(ebte+pr) + 0.6*pl
out = df.iloc[tei][['household_key','snapshot_day']].copy()
out['prediction'] = pred
path = agent_api.save_table(out, 'pred_e013.parquet')
print('saved', path, out.shape, 'pred range', round(pred.min(),1), round(pred.max(),1))
