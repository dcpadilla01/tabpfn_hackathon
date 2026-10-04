
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
print('saved shape', t.shape)
keys = ['household_key','snapshot_day']
feat_cols = [c for c in t.columns if c not in keys + ['index']]
print('n feature cols', len(feat_cols))
print('dtypes', dict(t.dtypes.value_counts()))
nonnum = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(t[c])]
print('nonnum count', len(nonnum), nonnum[:25])

tt = agent_api.train_targets()
print('targets', tt.shape)
print('y stats', {k: round(float(v),2) for k,v in tt['future_spend_4w'].describe().items()})
m = t.merge(tt, on=keys, how='inner')
print('merged', m.shape)
y = m['future_spend_4w'].astype(float).values
print('y mean %.2f median %.2f zero%% %.3f' % (y.mean(), np.median(y), (y==0).mean()))

Xdf = m[feat_cols].copy()
for c in nonnum:
    Xdf[c] = pd.to_numeric(Xdf[c], errors='coerce')
X = Xdf.values.astype(float)
mask = np.isnan(X)
cm = np.nanmean(X, axis=0)
X[mask] = np.take(cm, np.where(mask)[1])
sd = X.std(0); sd[sd==0] = 1
Xz = (X - cm)/sd

corr = np.zeros(len(feat_cols))
for j in range(len(feat_cols)):
    if sd[j] > 0:
        c = np.corrcoef(Xz[:,j], y)[0,1]
        corr[j] = 0 if np.isnan(c) else c
order = np.argsort(-np.abs(corr))
print('\ntop 45 by |corr with target| (train rows):')
for j in order[:45]:
    print('%-38s % .3f' % (feat_cols[j], corr[j]))

folds = m['snapshot_day'].values
days = sorted(set(folds))
print('\ntrain snapshot days:', days)
G = {}; B = {}; Xv = {}; yv = {}; trmask = {}
for d in days:
    tr = folds != d
    G[d] = Xz[tr].T @ Xz[tr]
    B[d] = Xz[tr].T @ y[tr]
    Xv[d] = Xz[~tr]; yv[d] = y[~tr]

def loso(idx, alpha, clip=True):
    idx = list(idx); errs = []
    for d in days:
        A = G[d][np.ix_(idx, idx)] + alpha*np.eye(len(idx))
        beta = np.linalg.solve(A, B[d][idx])
        p = Xv[d][:, idx] @ beta
        if clip: p = np.maximum(p, 0.0)
        errs.append(np.abs(p - yv[d]).mean())
    return float(np.mean(errs))

full = list(range(len(feat_cols)))
print('\nridge LOSO MAE by alpha (all %d feats):' % len(feat_cols))
best_a, best_m = None, 1e9
for a in [1, 3, 10, 30, 100, 300, 1000]:
    mm = loso(full, a)
    print('alpha %5d -> %.3f' % (a, mm))
    if mm < best_m: best_m, best_a = mm, a
print('best alpha', best_a, round(best_m,3))
print('mean-baseline LOSO MAE %.3f' % np.mean([np.abs(yv[d] - y[folds!=d].mean()).mean() for d in days]))

print('\nLOSO MAE by top-k |corr| features (alpha=%d):' % best_a)
for k in [5,10,20,40,60,80,120,180,250,320,len(feat_cols)]:
    print('k=%3d -> %.3f' % (k, loso(order[:k], best_a)))


# ---- cell ----
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
feat_cols = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values

Xdf = m[feat_cols].copy()
nonnum = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(Xdf[c])]
for c in nonnum:
    Xdf[c] = pd.to_numeric(Xdf[c], errors='coerce')
X = Xdf.values.astype(float)
keep = [j for j in range(len(feat_cols)) if not np.isnan(X[:,j]).all()]
feat_cols2 = [feat_cols[j] for j in keep]
X = X[:, keep]
mask = np.isnan(X)
cm = np.nanmean(X, axis=0)
X[mask] = np.take(cm, np.where(mask)[1])
sd = X.std(0); bad = sd <= 1e-12
Xz = (X - cm)/np.where(bad, 1, sd)

corr = np.nan_to_num(np.array([np.corrcoef(Xz[:,j], y)[0,1] if not bad[j] else 0.0 for j in range(len(feat_cols2))]))
order = np.argsort(-np.abs(corr))

folds = m['snapshot_day'].values.astype(int)
days = sorted(set(folds))
G={};B={};Xv={};yv={}
for d in days:
    tr = folds != d
    G[d]=Xz[tr].T@Xz[tr]; B[d]=Xz[tr].T@y[tr]; Xv[d]=Xz[~tr]; yv[d]=y[~tr]

def loso(idx, alpha, clip=True, ridge=True):
    idx=list(idx); errs=[]
    for d in days:
        A=G[d][np.ix_(idx,idx)]
        if ridge: A=A+alpha*np.eye(len(idx))
        beta=np.linalg.solve(A,B[d][idx])
        p=Xv[d][:,idx]@beta
        if clip: p=np.maximum(p,0.0)
        errs.append(np.abs(p-yv[d]).mean())
    return float(np.mean(errs))

full=list(range(len(feat_cols2)))
print('ridge LOSO MAE (all %d feats):'%len(feat_cols2))
best_a,best_m=None,1e9
for a in [3,10,30,100,300,1000,3000]:
    mm=loso(full,a)
    print('alpha %5d -> %.3f'%(a,mm))
    if mm<best_m: best_m,best_a=mm,a
print('best alpha',best_a,round(best_m,3))

print('\nLOSO MAE top-k |corr| (alpha=%d):'%best_a)
for k in [10,20,40,60,80,120,160,200,240,280,320,len(feat_cols2)]:
    print('k=%3d -> %.3f'%(k,loso(order[:k],best_a)))

print('\nLOSO MAE excluding bottom-k |corr| (alpha=%d):'%best_a)
for k in [0,20,40,60,80,100,140,180]:
    drop=set(order[-k:].tolist()) if k>0 else set()
    idx=[j for j in full if j not in drop]
    print('drop worst %3d -> n=%3d MAE %.3f'%(k,len(idx),loso(idx,best_a)))

# ---- cell ----
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
feat_cols = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values
folds = m['snapshot_day'].values.astype(int)

Xdf = m[feat_cols].copy()
nonnum = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(Xdf[c])]
for c in nonnum:
    Xdf[c] = pd.to_numeric(Xdf[c], errors='coerce')
X = Xdf.values.astype(float)
keep = [j for j in range(len(feat_cols)) if not np.isnan(X[:,j]).all()]
feat_cols2 = [feat_cols[j] for j in keep]
X = X[:, keep]
cm = np.nanmean(X, axis=0); mask = np.isnan(X)
X[mask] = np.take(cm, np.where(mask)[1])
sd = X.std(0); bad = sd <= 1e-12
Xz = (X - cm)/np.where(bad,1,sd)
corr = np.nan_to_num(np.array([np.corrcoef(Xz[:,j], y)[0,1] if not bad[j] else 0.0 for j in range(len(feat_cols2))]))
order = np.argsort(-np.abs(corr))

def fit_eval(train_mask, val_mask, idx, alpha, logt=False, clip=True):
    Xtr, ytr = Xz[train_mask][:, idx], y[train_mask]
    Xva, yva = Xz[val_mask][:, idx], y[val_mask]
    yt = np.log1p(ytr) if logt else ytr
    A = Xtr.T@Xtr + alpha*np.eye(len(idx))
    b = Xtr.T@yt
    beta = np.linalg.solve(A, b)
    p = Xva@beta
    if logt: p = np.expm1(p)
    if clip: p = np.clip(p, 0, None)
    return np.abs(p - yva).mean()

# time-aware: train on snapshots <= 375, validate on 403 & 431
trm = folds <= 375
print('per-day val MAE (full feats, ridge a=1000, raw):')
for d in [375, 403, 431]:
    vm = folds == d
    print(' day', d, 'n=%d ymean=%.1f MAE %.2f' % (vm.sum(), y[vm].mean(), fit_eval(trm, vm, list(range(len(feat_cols2))), 1000)))

vam = (folds == 403) | (folds == 431)
full = list(range(len(feat_cols2)))
print('\nTIME-AWARE proxy (train <=375, val {403,431}):')
print('%-28s %s' % ('variant','MAE'))
for a in [30, 100, 300, 1000, 3000]:
    print('ridge raw a=%-5d all      %.3f' % (a, fit_eval(trm, vam, full, a)))
for a in [30, 100, 300, 1000, 3000]:
    print('ridge log a=%-5d all      %.3f' % (a, fit_eval(trm, vam, full, a, logt=True)))
print('mean-baseline            %.3f' % np.abs(y[vam] - y[trm].mean()).mean())

print('\ntop-k |corr| raw a=1000:')
for k in [20,40,60,80,120,160,200,240,280,360]:
    print('k=%3d -> %.3f' % (k, fit_eval(trm, vam, order[:k], 1000)))
print('\ntop-k |corr| log a=1000:')
for k in [20,40,60,80,120,160,200,240,280,360]:
    print('k=%3d -> %.3f' % (k, fit_eval(trm, vam, order[:k], 1000, logt=True)))

# ---- cell ----
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
fc = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values
folds = m['snapshot_day'].values.astype(int)
print('mean y by day:', {int(d): round(float(y[folds==d].mean()),1) for d in sorted(set(folds))})

g = lambda c: m[c].astype(float).values
spend_28 = g('spend_28')
# find tlag cols
tlags = sorted([c for c in fc if c.startswith('tlag_') and c[5:].isdigit()], key=lambda c:int(c[5:]))
print('tlag cols:', tlags[:14])
W = np.column_stack([g(c) for c in ['spend_28']+tlags[:12]])  # windows t..t-11 (28d each)
med13 = np.nanmedian(W, axis=1)
mean13 = np.nanmean(W, axis=1)
ewma4 = g('ewma_4'); ewma8 = g('ewma_8'); ewma2 = g('ewma_2')
spend_91 = g('spend_91'); spend_182 = g('spend_182'); spend_364 = g('spend_364')
active = (W > 0).mean(1)

trm = folds <= 375; vam = (folds==403)|(folds==431)
def mae(p, mask=vam): return float(np.abs(np.clip(p,0,None)[mask] - y[mask]).mean())

print('\nstandalone predictor MAE on val {403,431}:')
cands = {'spend_28':spend_28,'ewma_2':ewma2,'ewma_4':ewma4,'ewma_8':ewma8,
         'mean13':mean13,'median13':med13,'spend_91':spend_91,'spend_182':spend_182,
         'spend_364':spend_364,'active*mean13':active*mean13,'active*ewma4':active*ewma4}
for k,v in cands.items(): print('  %-16s %.2f' % (k, mae(v)))
print('  mean-baseline   %.2f' % mae(np.full(len(y), y[trm].mean())))

# decay-weighted window average, w_i = decay^i
best=None
for dec in [1.0,0.9,0.8,0.7,0.6,0.5]:
    w = dec**np.arange(12); w/=w.sum()
    p = np.nansum(W*w,axis=1)
    print('  decay%.1f winavg  %.2f' % (dec, mae(p)))
    if best is None or mae(p)<best[1]: best=(dec,mae(p))
# blend grid: a*ewma4 + b*median13 + c*active*mean13 + d
A = np.column_stack([ewma4, med13, active*mean13, np.ones(len(y))])
from itertools import product
bestb=None
for a,b,c in product([0,.25,.5,.75,1],[0,.25,.5,.75,1],[0,.5,1]):
    if a+b+c>1.5: continue
    p = a*ewma4 + b*med13 + c*active*mean13 + (1-min(1,a+b+c))*y[trm].mean()*0  # no const
    mm = mae(p)
    if bestb is None or mm<bestb[1]: bestb=(a,b,c,mm)
print('  best blend a,b,c:', bestb)

# winsorization / concavity of best simple predictor
p = ewma4
for cap in [400,600,800,1200,2000]:
    print('  ewma4 capped@%d %.2f' % (cap, mae(np.minimum(p,cap))))
for pw in [0.5,0.7,0.85,1.0,1.15,1.3]:
    s = np.sign(p)*np.abs(p)**pw
    print('  ewma4 pow%.2f %.2f' % (pw, mae(s)))
# sqrt-domain blend fit on train via least squares (mean fit) then MAE eval
for cols,name in [([ewma4,med13],'e4+med'),([ewma4,med13,active*mean13],'e4+med+am'),([np.sqrt(ewma4),np.sqrt(med13)],'sqrt e4+med')]:
    Xm = np.column_stack(cols+[np.ones(len(y))])
    beta,*_ = np.linalg.lstsq(Xm[trm], y[trm], rcond=None)
    print('  LS %s: %.2f' % (name, mae(Xm@beta)))

# ---- cell ----
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
fc = [c for c in t.columns if c not in keys + ['index']]
print('col names containing tlag/ewma/spend/trip/day/week/active:')
import re
pat = re.compile(r'tlag|ewma|spend|trip|line|day|week|active|life|yoy|recen|gap|zero|wblk|ws_|wl_|wmean|sw_|rdecay|ts_')
print(sorted([c for c in fc if pat.search(c)])[:120])
print('\nALL cols (%d):' % len(fc))
print(fc)

# ---- cell ----
import agent_api, numpy as np, pandas as pd
from itertools import product

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
fc = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values
folds = m['snapshot_day'].values.astype(int)
trm = folds <= 375; vam = (folds==403)|(folds==431)
def mae(p): return float(np.abs(np.clip(p,0,None)[vam] - y[vam]).mean())

g = lambda c: m[c].astype(float).values
tlags = ['tlag_%d'%i for i in range(2,14)]
W = np.column_stack([g('spend_28')]+[g(c) for c in tlags])  # 13 windows: t..t-12
med13 = np.nanmedian(W, axis=1); mean13 = np.nanmean(W, axis=1)
ewma4 = g('ewma_4'); ewma2 = g('ewma_2'); ewma8 = g('ewma_8')
active = (W > 0).mean(1)
sp91 = g('spend_91'); sp84=g('spend_84'); sp112=g('spend_112')

print('decay-weighted window avg (13 windows):')
for dec in [1.0,0.9,0.8,0.7,0.6,0.5,0.4]:
    w = dec**np.arange(13); w/=w.sum()
    print('  decay%.1f -> %.2f' % (dec, mae(np.nansum(W*w,axis=1))))
print('trimmed mean (drop min/max of 13):')
Ws = np.sort(W, axis=1)
print('  %.2f' % mae(Ws[:,1:-1].mean(1)))
print('mean of top-6 windows: %.2f' % mae(Ws[:,-6:].mean(1)))
print('median of nonzero windows: %.2f' % mae(np.nanmedian(np.where(W>0,W,np.nan),axis=1)))

# LS blends on raw domain (fit on train only)
cands = {'e4':ewma4,'med':med13,'am':active*mean13,'m13':mean13,'e2':ewma2,'e8':ewma8}
names = list(cands)
Xall = np.column_stack([cands[n] for n in names] + [np.ones(len(y))])
best=None
for r in [1,2,3]:
    for combo in product(range(len(names)), repeat=r):
        if len(set(combo))<r: continue
        cols=[cands[names[i]] for i in combo]+[np.ones(len(y))]
        Xm=np.column_stack(cols)
        beta,*_=np.linalg.lstsq(Xm[trm], y[trm], rcond=None)
        mm=mae(Xm@beta)
        if best is None or mm<best[1]: best=(tuple(names[i] for i in combo), mm)
print('\nbest LS blend (raw):', best)
# sqrt domain LS
sq = lambda v: np.sqrt(np.clip(v,0,None))
best2=None
for r in [1,2,3]:
    for combo in product(range(len(names)), repeat=r):
        if len(set(combo))<r: continue
        Xm=np.column_stack([sq(cands[names[i]]) for i in combo]+[np.ones(len(y))])
        beta,*_=np.linalg.lstsq(Xm[trm], y[trm], rcond=None)
        mm=mae(Xm@beta)
        if best2 is None or mm<best2[1]: best2=(tuple(names[i] for i in combo), mm)
print('best LS blend (sqrt):', best2)

# fixed simple blends
print('\nfixed blends:')
for a,b,c in [(1,0,0),(0,1,0),(0,0,1),(.5,.5,0),(.5,0,.5),(0,.5,.5),(.34,.33,.33),(.6,.2,.2),(.4,.4,.2),(.2,.6,.2),(.2,.2,.6)]:
    print('  e4*%.2f+med*%.2f+am*%.2f -> %.2f' % (a,b,c, mae(a*ewma4+b*med13+c*active*mean13)))
# concavity on median13
for pw in [0.8,0.9,1.0,1.1,1.2]:
    print('  med13 pow%.1f -> %.2f' % (pw, mae(np.sign(med13)*np.abs(med13)**pw)))
for cap in [300,500,800]:
    print('  med13 cap%d -> %.2f' % (cap, mae(np.minimum(med13,cap))))

# ---- cell ----
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
fc = [c for c in t.columns if c not in keys + ['index']]
tt = agent_api.train_targets()
m = t.merge(tt, on=keys, how='inner')
y = m['future_spend_4w'].astype(float).values
folds = m['snapshot_day'].values.astype(int)

# verify tlag_k alignment vs weekly lags ws_1..ws_26 (ws_j = spend in week j back)
ws = np.column_stack([m['ws_%d'%j].astype(float).values for j in range(1,27)])
g = lambda c: m[c].astype(float).values
tl2 = g('tlag_2'); tl3 = g('tlag_3'); tl13 = g('tlag_13')
for combo,name in [((4,5,6,7),'ws5-8'),((5,6,7,8),'ws6-9'),((3,4,5,6),'ws4-7')]:
    s = ws[:,[j-1 for j in combo]].sum(1)
    print('corr(tlag_2, %s) = %.4f' % (name, np.corrcoef(tl2, s)[0,1]))
for combo,name in [((5,6,7,8),'ws6-9'),((6,7,8,9),'ws7-10')]:
    s = ws[:,[j-1 for j in combo]].sum(1)
    print('corr(tlag_3, %s) = %.4f' % (name, np.corrcoef(tl3, s)[0,1]))
s13 = ws[:, [12,13,14,15]].sum(1)  # weeks 13-16 back
print('corr(tlag_13, ws13-16) = %.4f' % np.corrcoef(tl13, s13)[0,1])
print('means: spend_28 %.1f tlag_2 %.1f tlag_3 %.1f tlag_13 %.1f' % (g('spend_28').mean(), tl2.mean(), tl3.mean(), tl13.mean()))

# build new summary predictors from existing as-of-safe columns
tlags = ['tlag_%d'%i for i in range(2,14)]  # windows t-1 .. t-12 (plus spend_28 = t)
W = np.column_stack([g('spend_28')]+[g(c) for c in tlags])  # 13 windows: most recent first
W = np.nan_to_num(W)
dec_avg = None
for dec,w in [(0.65, None)]:
    w = dec**np.arange(13); w/=w.sum()
    dec_avg = W@w
med13 = np.median(W, axis=1)
Ws = np.sort(W, axis=1)
trim = Ws[:,1:-1].mean(1)
active = (W>0).mean(1)
am = active*W.mean(1)
nzmed = np.where((W>0).any(1), np.median(np.where(W>0,W,np.nan),axis=1), 0)
# LS trend slope over 13 windows
tt_ = np.arange(13, dtype=float); tt_ = (tt_-tt_.mean())/tt_.std()
slope = (W*tt_).sum(1)
cv13 = W.std(1)/np.maximum(W.mean(1),1)
new = pd.DataFrame({'household_key':m['household_key'],'snapshot_day':m['snapshot_day'],
    'dec_avg65':dec_avg,'med13':med13,'trim13':trim,'act_mean13':am,'nzmed13':np.nan_to_num(nzmed),
    'slope13':slope,'cv13':cv13,'max13':W.max(1),'min13':W.min(1)})
print('\nnew feats corr with y:', {c: round(float(np.corrcoef(new[c],y)[0,1]),3) for c in new.columns[2:]})

trm = folds<=375; vam=(folds==403)|(folds==431)
def mae(p): return float(np.abs(np.clip(p,0,None)[vam]-y[vam]).mean())
print('proxy MAE dec_avg65 %.2f | med13 %.2f | trim %.2f | am %.2f' % (mae(dec_avg), mae(med13), mae(trim), mae(am)))
print('blend 0.5*dec+0.5*med %.2f | 0.7*dec+0.3*med %.2f' % (mae(0.5*dec_avg+0.5*med13), mae(0.7*dec_avg+0.3*med13)))
print('blend dec+am %.2f' % mae(0.5*dec_avg+0.5*am))

# ---- cell ----
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
Wcols = ['spend_28'] + ['tlag_%d'%i for i in range(2,13)]
W = t[Wcols].astype(float).values
W = np.nan_to_num(W)
ten = t['tenure_days'].astype(float).values
# window k (0-based) needs tenure >= 28*(k+1)
avail = (ten[:,None] >= 28*np.arange(1,14)[None,:])

feats = {}
for d in [0.5,0.6,0.65,0.7,0.8]:
    w = d**np.arange(13); w/=w.sum()
    feats['dec%d'%(d*100)] = W@w
    wv = np.where(avail, w[None,:], 0.0); wv = wv/np.maximum(wv.sum(1,keepdims=True),1e-9)
    feats['dec%d_avail'%(d*100)] = (W*wv).sum(1)
w = 0.65**np.arange(13); w/=w.sum(); dec65 = W@w
feats['lg_dec65'] = np.log1p(dec65)
feats['med13'] = np.median(W,axis=1)
Ws = np.sort(W,axis=1); feats['trim13'] = Ws[:,1:-1].mean(1)
tt_ = (np.arange(13)-6)/np.sqrt(143.0); feats['slope13'] = (W*tt_).sum(1)
feats['cv13'] = W.std(1)/np.maximum(W.mean(1),1.0)
feats['max13'] = W.max(1)
zs = (W==0); feats['zero_streak'] = (np.cumprod(zs,axis=1)).sum(1)  # leading consecutive zero windows
old = W[:,5:13].mean(1); feats['momentum'] = dec65/np.maximum(old,1.0)
feats['act13'] = (W>0).mean(1)
feats['mean13_avail'] = np.where(avail.any(1), (W*np.where(avail,1,0)).sum(1)/np.maximum(avail.sum(1),1), 0.0)
newdf = pd.DataFrame(np.column_stack([t[keys[0]], t[keys[1]]]+[feats[k] for k in feats]), columns=keys+list(feats))
for c in newdf.columns[2:]: newdf[c]=newdf[c].astype(float)
print('new feats:', len(feats))

# proxy check on train rows
ttg = agent_api.train_targets()
m = t.merge(ttg, on=keys)
mn = m.merge(newdf, on=keys, suffixes=('','_n'))
y = mn['future_spend_4w'].astype(float).values
folds = mn['snapshot_day'].values.astype(int)
trm = folds<=375; vam=(folds==403)|(folds==431)
def mae(p): return float(np.abs(np.clip(p,0,None)[vam]-y[vam]).mean())
for k in feats: print('  %-14s %.2f' % (k, mae(mn[k].values)))

# market-level seasonality quick look
snap = agent_api.snapshot()
tr = snap.transactions
wk = ((tr['day']+8)//7).astype(int)
g = tr.groupby(wk)['sales_value'].sum()
gidx = g.groupby(g.index//13).mean()  # 13-week blocks
print('\nretailer weekly spend by 13-wk block:'); print(gidx.round(0))
w2 = g.reindex(range(1,103)).fillna(0)
print('wk 5-19 mean %.0f | wk 57-71 mean %.0f | wk 31-45 mean %.0f' % (w2[5:20].mean(), w2[57:72].mean(), w2[31:46].mean()))
# yoy weekly ratio
r = w2/w2.shift(52)
print('yoy ratio wk 57-70:', r[57:71].round(2).tolist())
print('yoy ratio wk 66-78 (val window):', r[66:79].round(2).tolist())

# ---- cell ----
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
Wcols = ['spend_28'] + ['tlag_%d'%i for i in range(2,14)]  # 13 windows: t..t-12
W = np.nan_to_num(t[Wcols].astype(float).values)
ten = t['tenure_days'].astype(float).values
avail = (ten[:,None] >= 28*np.arange(1,14)[None,:])

feats = {}
for d in [0.5,0.6,0.65,0.7,0.8]:
    w = d**np.arange(13); w/=w.sum()
    feats['dec%d'%(d*100)] = W@w
    wv = np.where(avail, w[None,:], 0.0); wv = wv/np.maximum(wv.sum(1,keepdims=True),1e-9)
    feats['dec%d_avail'%(d*100)] = (W*wv).sum(1)
w = 0.65**np.arange(13); w/=w.sum(); dec65 = W@w
feats['lg_dec65'] = np.log1p(dec65)
feats['med13'] = np.median(W,axis=1)
Ws = np.sort(W,axis=1); feats['trim13'] = Ws[:,1:-1].mean(1)
tt_ = (np.arange(13)-6)/np.sqrt(143.0); feats['slope13'] = (W*tt_).sum(1)
feats['cv13'] = W.std(1)/np.maximum(W.mean(1),1.0)
feats['max13'] = W.max(1)
zs = (W==0); feats['zero_streak'] = (np.cumprod(zs,axis=1)).sum(1)
old = W[:,5:13].mean(1); feats['momentum'] = dec65/np.maximum(old,1.0)
feats['act13'] = (W>0).mean(1)
feats['mean13_avail'] = np.where(avail.any(1), (W*np.where(avail,1,0)).sum(1)/np.maximum(avail.sum(1),1), 0.0)
newdf = pd.DataFrame(np.column_stack([t[keys[0]].values, t[keys[1]].values]+[feats[k] for k in feats]), columns=keys+list(feats))
for c in newdf.columns[2:]: newdf[c]=newdf[c].astype(float)
print('new feats:', len(feats))

ttg = agent_api.train_targets()
mn = t.merge(ttg, on=keys).merge(newdf, on=keys)
y = mn['future_spend_4w'].astype(float).values
folds = mn['snapshot_day'].values.astype(int)
trm = folds<=375; vam=(folds==403)|(folds==431)
def mae(p): return float(np.abs(np.clip(p,0,None)[vam]-y[vam]).mean())
for k in feats: print('  %-14s %.2f' % (k, mae(mn[k].values)))

snap = agent_api.snapshot()
tr = snap.transactions
wk = ((tr['day']+8)//7).astype(int)
g = tr.groupby(wk)['sales_value'].sum()
w2 = g.reindex(range(1,103)).fillna(0)
print('\nretailer weekly spend: wk5-19 %.0f wk31-45 %.0f wk57-71 %.0f' % (w2[5:20].mean(), w2[31:46].mean(), w2[57:72].mean()))
r = w2/w2.shift(52)
print('yoy ratio wk57-70:', r[57:71].round(2).tolist())
print('yoy ratio wk66-78:', r[66:79].round(2).tolist())

# ---- cell ----
import agent_api, numpy as np, pandas as pd

snap = agent_api.snapshot()
tr = snap.transactions
wk = ((tr['day']+8)//7).astype(int)
g = tr.groupby(wk)['sales_value'].sum().reindex(range(1,103)).fillna(0)
print('weekly market spend, weeks 1-102:')
print(g.round(0).values)
# seasonal index: analog weeks (target - 52) vs (snapshot week - 52)
def idx(target_wk0, cur_wk0, span=4):
    # target window weeks target_wk0..target_wk0+3 (year2), analog year1 weeks -52
    a = g[[w-52 for w in range(target_wk0, target_wk0+4)]].sum()
    b = g[[w-52 for w in range(cur_wk0-3, cur_wk0+1)]].sum()
    return a/b
print('\nseasonal index (year1 analog of target window / year1 analog of current week):')
for sd in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    cw = (sd+8)//7; tw = (sd+29+8)//7  # first target week
    print('  snap %3d curwk %2d tgtwk %2d idx %.2f' % (sd, cw, tw, idx(tw, cw)))

# bias of dec65 by snapshot
t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys=['household_key','snapshot_day']
W = np.nan_to_num(t[['spend_28']+['tlag_%d'%i for i in range(2,14)]].astype(float).values)
w = 0.65**np.arange(13); w/=w.sum(); dec65 = W@w
ttg = agent_api.train_targets()
mn = t[['household_key','snapshot_day']].merge(ttg, on=keys)
mn['dec65'] = dec65
y = mn['future_spend_4w'].values; p = mn['dec65'].values
folds = mn['snapshot_day'].values.astype(int)
print('\ndec65 bias (mean pred - mean y) & MAE by snapshot:')
for d in sorted(set(folds)):
    mm = folds==d
    print('  %3d: pred %7.1f y %7.1f bias %6.1f MAE %6.1f' % (d, p[mm].mean(), y[mm].mean(), p[mm].mean()-y[mm].mean(), np.abs(p[mm]-y[mm]).mean()))

# ---- cell ----
import agent_api, numpy as np, pandas as pd

snap = agent_api.snapshot()
tr = snap.transactions
tr = tr[tr.day <= 459]
wk = ((tr['day']+8)//7).astype(int)  # 1..66
# stable households: active in both year1 (wk1-52) and year2 (wk53-66)
hh_wk = tr.groupby(['household_key', wk])['sales_value'].sum()
piv = hh_wk.unstack(fill_value=0.0)
y1 = piv[[w for w in range(1,53) if w in piv.columns]]
y2 = piv[[w for w in range(53,67) if w in piv.columns]]
act1 = (y1>0).sum(1); act2 = (y2>0).sum(1)
stable = piv[(act1>=20)&(act2>=10)]
print('households total %d, stable %d' % (len(piv), len(stable)))
s1 = stable[[w for w in range(1,53) if w in stable.columns]].mean(0)
s2 = stable[[w for w in range(53,67) if w in stable.columns]].mean(0)
s2.index = [w-52 for w in s2.index]
prof = pd.concat([s1, s2], axis=1).mean(1)  # per-household avg weekly spend by week-of-year 1..14
prof_full = s1.copy()
print('\nper-household avg weekly spend by week-of-year (year1), every 4 wks:')
print(prof_full.round(1).iloc[::4])
overall = prof_full.mean()
seas = prof_full/overall
print('\nseasonal index by week-of-year (year1), selected:')
for w in [1,5,9,13,17,21,25,29,33,37,41,45,49,52]:
    print('  wk %2d: %.2f' % (w, seas[w]))
# index for validation target windows: weeks 67-70 -> woy 15-18; current wks 63-66 -> woy 11-14
def sidx(woy_list): return seas[woy_list].mean()/seas.mean()
print('\nseasonal index target-window/current-window:')
for sd in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    cw = (sd+8)//7; tw = (sd+29+8)//7
    cw_y = [w-52 if w>52 else w for w in range(cw-3,cw+1)]
    tw_y = [w-52 if w>52 else w for w in range(tw,tw+4)]
    print('  snap %3d -> %.3f' % (sd, sidx(tw_y)/sidx(cw_y)))

# dec65 bias by snapshot
t = agent_api.load_saved('e009_ewma_longlags.parquet')
W = np.nan_to_num(t[['spend_28']+['tlag_%d'%i for i in range(2,14)]].astype(float).values)
w = 0.65**np.arange(13); w/=w.sum(); dec65 = W@w
ttg = agent_api.train_targets()
mn = t[['household_key','snapshot_day']].merge(ttg, on=['household_key','snapshot_day'])
mn['dec65'] = dec65
y = mn['future_spend_4w'].values; p = mn['dec65'].values
folds = mn['snapshot_day'].values.astype(int)
print('\ndec65 by snapshot: pred / y / MAE:')
for d in sorted(set(folds)):
    mm = folds==d
    print('  %3d: %7.1f %7.1f %6.1f' % (d, p[mm].mean(), y[mm].mean(), np.abs(p[mm]-y[mm]).mean()))

# ---- cell ----
import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys=['household_key','snapshot_day']
W = np.nan_to_num(t[['spend_28']+['tlag_%d'%i for i in range(2,14)]].astype(float).values)
ten = t['tenure_days'].astype(float).values
avail = (ten[:,None] >= 28*np.arange(1,14)[None,:])
feats={}
for d in [0.5,0.6,0.65,0.7,0.8]:
    w=d**np.arange(13); w/=w.sum()
    feats['dec%d'%(d*100)]=W@w
    wv=np.where(avail,w[None,:],0.0); wv=wv/np.maximum(wv.sum(1,keepdims=True),1e-9)
    feats['dec%d_avail'%(d*100)]=(W*wv).sum(1)
w=0.65**np.arange(13); w/=w.sum(); dec65=W@w
feats['lg_dec65']=np.log1p(dec65)
feats['med13']=np.median(W,axis=1)
Ws=np.sort(W,axis=1); feats['trim13']=Ws[:,1:-1].mean(1)
tt_=(np.arange(13)-6)/np.sqrt(143.0); feats['slope13']=(W*tt_).sum(1)
feats['cv13']=W.std(1)/np.maximum(W.mean(1),1.0)
feats['max13']=W.max(1)
feats['zero_streak']=(np.cumprod(W==0,axis=1)).sum(1)
feats['momentum']=dec65/np.maximum(W[:,5:13].mean(1),1.0)
feats['act13']=(W>0).mean(1)
feats['mean13_avail']=np.where(avail.any(1),(W*np.where(avail,1,0)).sum(1)/np.maximum(avail.sum(1),1),0.0)
newdf=pd.DataFrame(np.column_stack([t[keys[0]].values,t[keys[1]].values]+[feats[k] for k in feats]),columns=keys+list(feats))
for c in newdf.columns[2:]: newdf[c]=newdf[c].astype(float)
full=t.merge(newdf,on=keys)
print('full table', full.shape)
path=agent_api.save_table(full,'e016_dec_predictors.parquet')
print('saved',path)

ttg=agent_api.train_targets()
mn=full.merge(ttg,on=keys)
y=mn['future_spend_4w'].values; p=mn['dec65'].values
folds=mn['snapshot_day'].values.astype(int)
print('\ndec65 by snapshot: pred / y / MAE:')
for d in sorted(set(folds)):
    mm=folds==d
    print('  %3d: %7.1f %7.1f %6.1f' % (d,p[mm].mean(),y[mm].mean(),np.abs(p[mm]-y[mm]).mean()))
trm=folds<=375; vam=(folds==403)|(folds==431)
print('proxy MAE dec65 %.2f (train<=375,val 403+431)' % np.abs(np.clip(p,0,None)[vam]-y[vam]).mean())