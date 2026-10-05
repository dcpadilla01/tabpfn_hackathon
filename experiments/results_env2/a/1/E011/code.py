
import pandas as pd, numpy as np

names = ['repro_e5','e005_preds','e005_newfeats','e004_features','e004_new','allF',
         'lagfeats','lagfeats2','f_weekly','e002_features',
         'e001_preds','e003_preds','e004_preds','e007_preds','e008_preds','e009_preds','e010_preds']
store = {}
for n in names:
    try:
        df = agent_api.load_saved(n + '.parquet')
        store[n] = df
        cols = list(df.columns)
        print('==', n, df.shape, 'ncols', len(cols))
        print('   ', cols[:18], '...' if len(cols) > 18 else '')
    except Exception as e:
        print('==', n, 'ERR', repr(e))

tt = agent_api.train_targets()
print('\ntrain_targets', tt.shape)
print(tt.future_spend_4w.describe())
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median'])
g['zero_share'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda x: (x <= 0).mean())
print(g)

r, p = store.get('repro_e5'), store.get('e005_preds')
if r is not None and p is not None:
    print('\nrepro head:'); print(r.head(3))
    print('e005_preds head:'); print(p.head(3))


# ---- cell ----

import pandas as pd, numpy as np
F = agent_api.load_saved('allF.parquet')
print('allF cols:'); print(list(F.columns))
NF = agent_api.load_saved('e005_newfeats.parquet')
print('\ne005_newfeats cols:'); print(list(NF.columns))
print('\nallF contains NF cols?', set(NF.columns)-set(F.columns))

# blend weights of e005 from repro p13/p11
r = agent_api.load_saved('repro_e5.parquet'); p = agent_api.load_saved('e005_preds.parquet')
m = r.merge(p, on=['household_key','snapshot_day'])
A = np.vstack([m.p13, m.p11, np.ones(len(m))]).T
w, res, *_ = np.linalg.lstsq(A, m.prediction, rcond=None)
print('\nblend coefs (p13,p11,intercept):', w, 'resid max', np.abs(A@w - m.prediction).max())

# differences among saved preds
preds = {n: agent_api.load_saved(n+'_preds.parquet').set_index(['household_key','snapshot_day'])['prediction'] for n in ['e003','e004','e005','e007','e008','e009','e010','e001']}
base = preds['e005']
for n, s in preds.items():
    d = (s - base).abs()
    print(n, 'mean|diff vs e005| = %.3f  corr=%.4f' % (d.mean(), np.corrcoef(s, base)[0,1]))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time
print('xgb', xgb.__version__)
r = agent_api.load_saved('repro_e5.parquet'); p = agent_api.load_saved('e005_preds.parquet')
m = r.merge(p, on=['household_key','snapshot_day'])
for a in [0.6,0.65,0.7,0.75,0.8]:
    pred = a*m.p13 + (1-a)*m.p11
    print('a=%.2f maxresid %.4f meanresid %.4f' % (a, np.abs(pred-m.prediction).max(), (pred-m.prediction).mean()))
# maybe blend includes clipping
pred = 0.7*m.p13+0.3*m.p11
print('clip200?', np.abs(np.minimum(pred,200)-m.prediction).mean())
print(m.prediction.describe())

F = agent_api.load_saved('allF.parquet')
print('allF dtypes non-numeric:', [c for c in F.columns if F[c].dtype==object])
print('NaN frac overall:', F.isna().mean().mean())
tt = agent_api.train_targets()
tr = F[F.snapshot_day.isin(tt.snapshot_day.unique())]
print('train rows in allF:', len(tr), 'target match?', np.allclose(tr.set_index(['household_key','snapshot_day']).sort_index()['future_spend_4w'].values,
      tt.set_index(['household_key','snapshot_day']).sort_index()['future_spend_4w'].values))
val = F[~F.snapshot_day.isin(tt.snapshot_day.unique())]
print('val rows:', len(val), 'val target nan?', val.future_spend_4w.isna().all())
print('val snapshot days:', sorted(val.snapshot_day.unique()))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')

F = agent_api.load_saved('allF.parquet')
TARGET = 'future_spend_4w'
DROP = [TARGET]
FEATS = [c for c in F.columns if c not in DROP]
TRAIN_DAYS = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def decay_weights(days, half=140):
    age = (431 - days).astype(float)
    return 0.5 ** (age / half)

def fit_xgb(Xtr, ytr, wtr, params, num_rounds):
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
    bst = xgb.train(params, dtr, num_boost_round=num_rounds, verbose_eval=False)
    return bst

def run_pair(feats_df, params, half=140, rounds=2400, lr=0.03, days_train_a=None, days_train_b=None, verbose=True):
    feats_df = feats_df.copy()
    y = feats_df[TARGET].values
    X = feats_df[FEATS].astype(float)
    days = feats_df['snapshot_day']
    dA = days_train_a or [d for d in TRAIN_DAYS if d <= 403]
    dB = days_train_b or [d for d in TRAIN_DAYS if d <= 375]
    pa = dict(params); pa['learning_rate'] = lr
    t0=time.time()
    mA = fit_xgb(X[days.isin(dA)], y[days.isin(dA)], decay_weights(days[days.isin(dA)], half), pa, rounds)
    mB = fit_xgb(X[days.isin(dB)], y[days.isin(dB)], decay_weights(days[days.isin(dB)], half), pa, rounds)
    out = {}
    # eval A on 431, B on 403
    for mdl, ev in [(mA,431),(mB,403)]:
        m = ev==431
        idx = days==ev
        p = mdl.predict(xgb.DMatrix(X[idx]))
        t = y[idx]
        out[ev] = np.abs(p-t).mean()
    if verbose:
        print('  eval MAE: d431=%.3f d403=%.3f proxy=%.3f (%.0fs)' % (out[431], out[403], 0.5*(out[431]+out[403]), time.time()-t0))
    return out, (mA,mB)

base_params = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=6,
                   min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
                   tree_method='hist', eval_metric=['mae'])
res, mdls = run_pair(F, base_params)


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
TRAIN_DAYS=[95,123,151,179,207,235,263,291,319,347,375,403,431]
base_params = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=6,
                   min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
                   tree_method='hist', eval_metric=['mae'])

def proxy(train_on_log=False, half=140, rounds=2400, lr=0.03, params=None, feats=None, seed=0):
    P = dict(base_params); P.update(params or {}); P['learning_rate']=lr
    if seed: P['seed']=seed
    df = F if feats is None else feats
    X = df[FEATS if feats is None else [c for c in feats.columns if c not in ('household_key','snapshot_day',TARGET)]].astype(float)
    y_raw = df[TARGET].values; y = np.log1p(y_raw) if train_on_log else y_raw
    days = df['snapshot_day']
    out={}
    for tr_max, ev in [(403,431),(375,403)]:
        m = days<=tr_max
        w = 0.5**((tr_max-days[m]).astype(float)/half)
        bst = xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
        idx = days==ev
        p = bst.predict(xgb.DMatrix(X[idx]))
        if train_on_log: p = np.expm1(p)
        out[ev]=np.abs(p-y_raw[idx]).mean()
    print('  d431=%.3f d403=%.3f proxy=%.3f' % (out[431],out[403],0.5*(out[431]+out[403])))
    return out

print('linear target (ref):'); proxy()
print('log1p target:'); proxy(train_on_log=True)


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
P = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=6, min_child_weight=10,
         subsample=0.8, colsample_bytree=0.7, tree_method='hist', learning_rate=0.03, eval_metric=['mae'])
days = F['snapshot_day']; X=F[FEATS].astype(float); y=F[TARGET].values
m = days<=403
w = 0.5**((403-days[m]).astype(float)/140)
bst = xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), 2400, verbose_eval=False)
idx = days==431
p = bst.predict(xgb.DMatrix(X[idx])); t = y[idx]
print('overall MAE %.3f  bias %+.3f' % (np.abs(p-t).mean(), (p-t).mean()))
bins=[0,1,25,75,150,300,600,1e9]
for lo,hi in zip(bins[:-1],bins[1:]):
    sel=(t>=lo)&(t<hi)
    if sel.sum()==0: continue
    print('t in [%6.0f,%6.0f): n=%4d  pred_mean=%7.1f  truth_mean=%7.1f  MAE=%7.1f  share_of_total_loss=%.2f' %
          (lo,hi,sel.sum(),p[sel].mean(),t[sel].mean(),np.abs(p[sel]-t[sel]).mean(),np.abs(p[sel]-t[sel]).sum()/np.abs(p-t).sum()))
# zero rows
sel = t==0
print('zero rows: n=%d pred mean %.1f  MAE contribution %.2f' % (sel.sum(), p[sel].mean(), np.abs(p[sel]-t[sel]).sum()/np.abs(p-t).sum()))
# top decile
q=np.quantile(t,0.9); sel=t>=q
print('top-decile rows: MAE=%.1f share=%.2f  (n=%d)' % (np.abs(p[sel]-t[sel]).mean(), np.abs(p[sel]-t[sel]).sum()/np.abs(p-t).sum(), sel.sum()))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']

def train_eval(params, tr_max=403, ev=431, rounds=2400, half=140, also_insample=False, lr=0.03):
    P=dict(params); P['learning_rate']=lr
    m=days<=tr_max
    w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    idx=days==ev; p=bst.predict(xgb.DMatrix(X[idx])); t=y[idx]
    mae=np.abs(p-t).mean()
    line='MAE=%.3f bias=%+.2f' % (mae,(p-t).mean())
    if also_insample:
        pi=bst.predict(xgb.DMatrix(X[m])); ti=y[m]
        line+='  | in-sample bias=%+.2f' % (pi-ti).mean()
        hi=ti>=np.quantile(ti,0.9)
        line+=' insample-top10 MAE=%.1f (pred %.0f vs true %.0f)' % (np.abs(pi[hi]-ti[hi]).mean(), pi[hi].mean(), ti[hi].mean())
    hi=t>=np.quantile(t,0.9)
    line+='  | top10: pred %.0f true %.0f MAE %.1f' % (p[hi].mean(), t[hi].mean(), np.abs(p[hi]-t[hi]).mean())
    print(line)
    return mae, bst

P50=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'])
t0=time.time()
print('ref a=.5 (insample check):'); train_eval(P50, also_insample=True)
print('a=.55:'); train_eval({**P50,'quantile_alpha':0.55})
print('a=.60:'); train_eval({**P50,'quantile_alpha':0.60})
print('depth8 mcw5:'); train_eval({**P50,'max_depth':8,'min_child_weight':5})
print('%.0fs' % (time.time()-t0))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
P50=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,colsample_bytree=0.7,tree_method='hist',learning_rate=0.03,eval_metric=['mae'])

def get_preds(alpha, tr_max=403, ev=431, rounds=2400, half=140, seed=0):
    P=dict(P50); P['quantile_alpha']=alpha; P['seed']=seed
    m=days<=tr_max
    w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    return bst.predict(xgb.DMatrix(X[days==ev])), y[days==ev]

t0=time.time()
p50,t = get_preds(0.5); p60,_ = get_preds(0.6); p70,_ = get_preds(0.7)
def ev_mae(p, t): return np.abs(p-t).mean()
print('p50 %.3f | p60 %.3f | p70 %.3f' % (ev_mae(p50,t), ev_mae(p60,t), ev_mae(p70,t)))
# conditional blend: s = sigmoid of p50 around threshold
for thr in [100,150,200,300]:
    for wgt in [0.3,0.5,0.8,1.0]:
        s = np.clip((p50-thr)/thr, 0, 1)*wgt   # 0 below thr, ->wgt far above
        p = p50 + s*(p60-p50)
        print('thr=%3d w=%.1f: MAE=%.3f (top10 %.1f)' % (thr,wgt,ev_mae(p,t),np.abs(p[t>=np.quantile(t,0.9)]-t[t>=np.quantile(t,0.9)]).mean()))
# simple global blends for reference
for a in [0.2,0.3]:
    print('global p50+(%.1f)(p60-p50): %.3f' % (a, ev_mae(p50+a*(p60-p50),t)))
print('%.0fs'%(time.time()-t0))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(max_depth=6,min_child_weight=10,subsample=0.8,colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03)

def mkp(obj, alpha=None, seed=0):
    P=dict(BASE); P['objective']=obj; P['seed']=seed
    if alpha is not None: P['quantile_alpha']=alpha
    return P

def preds(P, tr_max=403, ev=431, rounds=2400, half=140):
    m=days<=tr_max; w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    return bst.predict(xgb.DMatrix(X[days==ev])), y[days==ev]

t0=time.time()
p50,t = preds(mkp('reg:quantileerror',0.5))
pm,_  = preds(mkp('reg:squarederror'))
print('p50 %.3f | mean-model %.3f | corr %.4f' % (np.abs(p50-t).mean(), np.abs(pm-t).mean(), np.corrcoef(p50,pm)[0,1]))
for lam in [0.1,0.2,0.3,0.4,0.5,0.7]:
    p = p50+lam*(pm-p50)
    hi=t>=np.quantile(t,0.9)
    print('lam=%.1f: MAE=%.3f top10=%.1f' % (lam, np.abs(p-t).mean(), np.abs(p[hi]-t[hi]).mean()))
# two-stage residual model
m=days<=403; w=0.5**((403-days[m]).astype(float)/140)
dtr=xgb.DMatrix(X[m],label=y[m],weight=w); bst50=xgb.train(mkp('reg:quantileerror',0.5),dtr,2400,verbose_eval=False)
resid = y[m]-bst50.predict(dtr)
dres=xgb.DMatrix(X[m],label=resid,weight=w)
for rr in [600,1200]:
    bstr=xgb.train(mkp('reg:squarederror'),dres,rr,verbose_eval=False)
    p2 = p50 + bstr.predict(xgb.DMatrix(X[days==431]))
    hi=t>=np.quantile(t,0.9)
    print('resid2stage rounds=%d: MAE=%.3f top10=%.1f' % (rr,np.abs(p2-t).mean(),np.abs(p[hi]-t[hi]).mean()))
print('%.0fs'%(time.time()-t0))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
          colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03)

def one_model(P, tr_max, ev, rounds=2400, half=140):
    m=days<=tr_max; w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    return bst.predict(xgb.DMatrix(X[days==ev]))

t0=time.time()
# single seed
p1 = one_model(dict(BASE,seed=0), 403, 431); t=y[days==431]
print('single seed: MAE=%.3f' % np.abs(p1-t).mean())
# 8-seed ensemble, same config
P8=[]
for s in range(8):
    P8.append(one_model(dict(BASE,seed=s), 403, 431))
p8=np.mean(P8,axis=0)
print('8-seed: MAE=%.3f (spread of members: %.3f-%.3f)' % (np.abs(p8-t).mean(), min(np.abs(q-t).mean() for q in P8), max(np.abs(q-t).mean() for q in P8)))
# 4-seed x 2 configs (subsample/colsample jitter)
P16=[]
for s in range(4):
    for (ss,cs,md,mcw) in [(0.8,0.7,6,10),(0.7,0.6,6,8)]:
        P16.append(one_model(dict(BASE,seed=s,subsample=ss,colsample_bytree=cs,max_depth=md,min_child_weight=mcw),403,431))
p16=np.mean(P16,axis=0)
print('8-mixed: MAE=%.3f' % np.abs(p16-t).mean())
print('%.0fs'%(time.time()-t0))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
          colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03)
def one_model(P, tr_max, ev, rounds=1200, half=140):
    m=days<=tr_max; w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    return bst.predict(xgb.DMatrix(X[days==ev]))
t0=time.time()
p1 = one_model(dict(BASE,seed=0),403,431); t=y[days==431]
print('1200r single: MAE=%.3f' % np.abs(p1-t).mean())
P8=[one_model(dict(BASE,seed=s),403,431) for s in range(8)]
p8=np.mean(P8,axis=0)
print('1200r 8-seed: MAE=%.3f  member MAEs: %s' % (np.abs(p8-t).mean(), ' '.join('%.2f'%np.abs(q-t).mean() for q in P8)))
print('%.0fs'%(time.time()-t0))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
tr = days<=403
# noise floor: correlations of target with trailing windows on train rows
for c in ['spend_28','spend_56','spend_84','spend_364','s1','s2','s3','s4','seq_mean']:
    print('%-10s corr=%.3f' % (c, np.corrcoef(F.loc[tr,c], y[tr])[0,1]))
# anchor MAEs on 431
idx=days==431; t=y[idx]
for c in ['spend_28','spend_84','seq_mean']:
    a=F.loc[idx,c].values
    print('anchor %-9s MAE=%.2f' % (c, np.abs(a-t).mean()))
# train p50 model
P=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
       colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03,seed=0)
m=days<=403; w=0.5**((403-days[m]).astype(float)/140)
bst=xgb.train(P,xgb.DMatrix(X[m],label=y[m],weight=w),1200,verbose_eval=False)
p=bst.predict(xgb.DMatrix(X[idx]))
print('model MAE=%.3f' % np.abs(p-t).mean())
s28=F.loc[idx,'spend_28'].values; s84=F.loc[idx,'spend_84'].values
for lam in [0.05,0.1,0.2,0.3,0.5]:
    for name,a in [('s28',s28),('s84',s84)]:
        q=p+lam*(a-p)
        print('blend lam=%.2f %-4s: MAE=%.3f' % (lam,name,np.abs(q-t).mean()))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
          colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03,seed=0)

def ev(cfg, tr_max, evd, rounds=1200, half=140):
    P=dict(BASE); P.update(cfg)
    m=days<=tr_max; w=0.5**((tr_max-days[m]).astype(float)/half)
    bst=xgb.train(P, xgb.DMatrix(X[m],label=y[m],weight=w), rounds, verbose_eval=False)
    idx=days==evd; p=bst.predict(xgb.DMatrix(X[idx]))
    return np.abs(p-y[idx]).mean()

t0=time.time()
print('ref (tr403,ev431): %.3f' % ev({},403,431))
print('tr375,ev431      : %.3f  <-- training-window effect' % ev({},375,431))
print('tr403,ev403? n/a')
print('mcw=5 : %.3f' % ev({'min_child_weight':5},403,431))
print('mcw=20: %.3f' % ev({'min_child_weight':20},403,431))
print('depth5: %.3f' % ev({'max_depth':5},403,431))
print('depth7: %.3f' % ev({'max_depth':7},403,431))
print('half100: %.3f' % ev({},403,431,half=100))
print('half200: %.3f' % ev({},403,431,half=200))
print('l1=1  : %.3f' % ev({'alpha':1.0},403,431))
print('l1=5  : %.3f' % ev({'alpha':5.0},403,431))
print('maxbin=256: %.3f' % ev({'max_bin':256},403,431))
print('cs=0.8: %.3f' % ev({'colsample_bytree':0.8},403,431))
print('ss=0.9: %.3f' % ev({'subsample':0.9},403,431))
print('tweedie1.3: %.3f' % ev({'objective':'reg:tweedie','tweedie_variance_power':1.3},403,431))
print('tweedie1.6: %.3f' % ev({'objective':'reg:tweedie','tweedie_variance_power':1.6},403,431))
print('%.0fs'%(time.time()-t0))


# ---- cell ----

import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('allF.parquet')
TARGET='future_spend_4w'; FEATS=[c for c in F.columns if c!=TARGET]
X=F[FEATS].astype(float); y=F[TARGET].values; days=F['snapshot_day']
BASE=dict(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=6,min_child_weight=10,subsample=0.8,
          colsample_bytree=0.7,tree_method='hist',eval_metric=['mae'],learning_rate=0.03)
t0=time.time()
m=days<=403
w=0.5**((403-days[m]).astype(float)/140)
dtr=xgb.DMatrix(X[m],label=y[m],weight=w)
seeds=[]
for s in range(8):
    bst=xgb.train(dict(BASE,seed=s), dtr, 1200, verbose_eval=False)
    seeds.append(bst.predict(xgb.DMatrix(X)))
print('8 models trained (%.0fs)'%(time.time()-t0))
p50 = np.mean(seeds,axis=0)
val = days>=459
s28 = F.loc[val,'spend_28'].values
p_final = 0.9*p50[val] + 0.1*s28
sub = F.loc[val,['household_key','snapshot_day']].copy()
sub['prediction']=p_final
print('val rows:',len(sub),'days:',sorted(sub.snapshot_day.unique()),'nan:',sub.prediction.isna().sum())
print('pred stats: mean %.1f med %.1f p90 %.1f max %.1f' % (p_final.mean(),np.median(p_final),np.quantile(p_final,0.9),p_final.max()))
e5 = agent_api.load_saved('e005_preds.parquet').set_index(['household_key','snapshot_day'])['prediction']
mm = sub.set_index(['household_key','snapshot_day']).join(e5.rename('e5'))
print('vs E005: corr %.4f  mean|diff| %.2f  mean(diff) %+.2f' % (np.corrcoef(mm.prediction,mm.e5)[0,1],(mm.prediction-mm.e5).abs().mean(),(mm.prediction-mm.e5).mean()))
path = agent_api.save_table(sub, 'e011_preds.parquet')
print('saved:', path)
