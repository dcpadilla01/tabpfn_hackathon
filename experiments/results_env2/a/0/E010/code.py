
import agent_api, pandas as pd, numpy as np

feats = agent_api.load_saved('feats_v3.parquet')
print("feats_v3 shape:", feats.shape)
print("columns:", list(feats.columns))

tt = agent_api.train_targets()
print("\ntrain_targets shape:", tt.shape)
m = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w']
print("\ntarget describe:\n", y.describe())
print("zero fraction:", (y==0).mean(), " zeros:", (y==0).sum(), "of", len(y))
print("quantiles:", np.percentile(y, [50,75,90,95,99]))

pred7 = agent_api.load_saved('pred_e007.parquet')
print("\npred_e007 shape:", pred7.shape, "cols:", list(pred7.columns))
print(pred7.head(3))
print("pred range:", pred7['prediction'].min(), pred7['prediction'].max())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats.drop(columns=['index']), on=['household_key','snapshot_day'], how='left')

TRAIN_DAYS = [95,123,151,179,207,235,263,291,319,347,375,403,431]
print("target mean/median by snapshot day:")
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).round(1))

FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]
print("\nn features:", len(FE))

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

tr = df[df.snapshot_day<=403]
va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xva,yva = make_xy(va)
print("train rows:", len(tr), "val rows:", len(va), "val zero frac:", (yva==0).mean())

def fit_q(Xtr,ytr,depth=4,mcw=20,nest=600,lr=0.05):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
                         max_depth=depth, min_child_weight=mcw, n_estimators=nest,
                         learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                         n_jobs=4, random_state=0)
    m.fit(Xtr,ytr)
    return m

m = fit_q(Xtr,ytr)
p = m.predict(Xva)
blend = Xva['exp4w_blend'].values
final = np.clip(0.7*p + 0.3*blend, 0, None)
print("\nlocal val (431): model-only MAE:", np.abs(p-yva).mean())
print("local val blend0.7 MAE:", np.abs(final-yva).mean())
print("persistence-only MAE:", np.abs(blend-yva).mean())
print("mean pred:", final.mean(), "mean y:", yva.mean())


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xva,yva = make_xy(va)

m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4,
                     min_child_weight=20, n_estimators=600, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0)
m.fit(Xtr,ytr)
p = m.predict(Xva)
blend = Xva['exp4w_blend'].values
final = np.clip(0.7*p+0.3*blend,0,None)

e = np.abs(final-yva)
print("MAE by target bucket:")
bins=[0,1,50,100,200,400,800,10000]
lab=pd.cut(yva,bins)
tmp=pd.DataFrame({'y':yva,'e':e,'p':final,'dsl':Xva['days_since_last'].values,'s84':Xva['spend_84'].values})
print(tmp.groupby(lab,observed=True).agg(n=('e','size'),mae=('e','mean'),mean_pred=('p','mean'),mean_y=('y','mean'),sum_e=('e','sum')).round(1))
print("\nshare of total abs error by bucket (%):")
print((tmp.groupby(lab,observed=True)['sum_e'].sum()/e.sum()*100).round(1))

print("\nMAE by days_since_last bucket:")
tmp['dsl_b']=pd.cut(tmp.dsl,[0,7,14,28,56,100,10000])
print(tmp.groupby('dsl_b',observed=True).agg(n=('e','size'),mae=('e','mean'),mean_pred=('p','mean'),mean_y=('y','mean')).round(1))

# how do zeros behave
z = tmp.y==0
print("\nzeros: n=",z.sum()," mean pred on zeros:",tmp.p[z].mean().round(1)," MAE on zeros:",tmp.e[z].mean().round(1))
nz = tmp.y>0
print("nonzeros: n=",nz.sum()," mean pred:",tmp.p[nz].mean().round(1)," mean y:",tmp.y[nz].mean().round(1)," MAE:",tmp.e[nz].mean().round(1))

# tune blend weight locally
for w in [0.5,0.6,0.7,0.8,0.9,1.0]:
    f=np.clip(w*p+(1-w)*blend,0,None)
    print(f"w={w}: MAE={np.abs(f-yva).mean():.3f}")


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xva,yva = make_xy(va)
blend = Xva['exp4w_blend'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0)
    m.fit(Xtr,ytr); return m

m = qmodel(Xtr,ytr); p = m.predict(Xva)
base = np.clip(0.7*p+0.3*blend,0,None)
print("base MAE:", np.abs(base-yva).mean())

tmp = pd.DataFrame({'y':yva,'e':np.abs(base-yva),'p':base,'dsl':Xva['days_since_last'].values})
tmp['dsl_b']=pd.cut(tmp.dsl,[0,7,14,28,56,100,10000])
print("\nMAE by days_since_last:")
print(tmp.groupby('dsl_b',observed=True).agg(n=('e','size'),mae=('e','mean'),mean_pred=('p','mean'),mean_y=('y','mean')).round(1))
z = tmp.y==0
print("\nzeros: n=",z.sum()," mean pred:",round(tmp.p[z].mean(),1)," MAE:",round(tmp.e[z].mean(),1)," err share:",round(tmp.e[z].sum()/tmp.e.sum()*100,1))

# --- idea A: two-part model: P(y>0) * median(y|y>0)
from xgboost import XGBClassifier
ycls = (ytr>0).astype(int)
clf = XGBClassifier(max_depth=4, min_child_weight=20, n_estimators=400, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, eval_metric='logloss')
clf.fit(Xtr,ycls)
pz = clf.predict_proba(Xva)[:,1]
m2 = qmodel(Xtr[ytr>0],ytr[ytr>0])
p2 = m2.predict(Xva)
two = np.clip(0.7*(pz*p2)+0.3*blend,0,None)
print("\nA two-part MAE:", np.abs(two-yva).mean())
for w in [0.5,0.7,0.9]:
    print(f"  two-part w={w}:", np.abs(np.clip(w*(pz*p2)+(1-w)*blend,0,None)-yva).mean())

# --- idea B: quantile on log(y+1), then expm1
m3 = qmodel(Xtr,np.log1p(ytr))
p3 = np.expm1(m3.predict(Xva))
print("\nB log-quantile MAE (w=0.7):", np.abs(np.clip(0.7*p3+0.3*blend,0,None)-yva).mean())
for w in [0.5,0.7,0.9,1.0]:
    print(f"  log w={w}:", np.abs(np.clip(w*p3+(1-w)*blend,0,None)-yva).mean())

# --- idea C: deeper/wider model to capture tail
m4 = qmodel(Xtr,ytr,depth=6,mcw=10,nest=900)
p4 = m4.predict(Xva)
print("\nC depth6 MAE:", np.abs(np.clip(0.7*p4+0.3*blend,0,None)-yva).mean())


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.isotonic import IsotonicRegression

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0,obj='reg:quantileerror'):
    kw = dict(objective=obj, max_depth=depth, min_child_weight=mcw, n_estimators=nest,
              learning_rate=lr, subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    if obj=='reg:quantileerror': kw['quantile_alpha']=alpha
    m = xgb.XGBRegressor(**kw); m.fit(Xtr,ytr); return m

tr375 = df[df.snapshot_day<=375]; tr403 = df[df.snapshot_day<=403]
cal403 = df[df.snapshot_day==403]; va431 = df[df.snapshot_day==431]
X375,y375 = make_xy(tr375); Xc,yc = make_xy(cal403); Xv,yv = make_xy(va431)
blv = Xv['exp4w_blend'].values; blc = Xc['exp4w_blend'].values

# honest calibration test: train<=375, calibrate on 403, evaluate on 431
m = qmodel(X375,y375)
pc = m.predict(Xc); pv = m.predict(Xv)
iso = IsotonicRegression(out_of_bounds='clip', y_min=0)
iso.fit(pc, yc)
pv_iso = iso.predict(pv)
print("honest iso-calib: raw MAE(431):", np.abs(np.clip(0.7*pv+0.3*blv,0,None)-yv).mean(),
      " iso MAE:", np.abs(np.clip(0.7*pv_iso+0.3*blv,0,None)-yv).mean())
# linear recalibration on cal set
a,b = np.polyfit(pc,yc,1)
print("linear recal a,b:",round(a,3),round(b,1), " MAE:", np.abs(np.clip(0.7*(a*pv+b)+0.3*blv,0,None)-yv).mean())

# --- OOF top-end boost: fit g on train OOF (train<=403, eval 431)
tr = df[df.snapshot_day<=403]; Xtr,ytr = make_xy(tr)
# OOF by snapshot
oof = np.zeros(len(tr))
for s in sorted(tr.snapshot_day.unique()):
    itr = tr[tr.snapshot_day!=s]; it = tr[tr.snapshot_day==s]
    Xi,yi = make_xy(itr); Xt,yt = make_xy(it)
    mm = qmodel(Xi,yi); oof[tr.index.get_indexer(it.index)] = mm.predict(Xt)
oof = np.clip(0.7*oof+0.3*tr['exp4w_blend'].values,0,None)
yo = tr['future_spend_4w'].values
# top-end boost: p' = p + g*max(0,p-T)
for T in [200,300,400]:
    x = np.maximum(0,oof-T)
    g = np.clip(np.polyfit(x, yo-oof, 1)[0], 0, 0.5)
    pv2 = np.clip(0.7*pv+0.3*blv,0,None)
    print(f"top-end boost T={T}: g={g:.3f}, MAE:", np.abs((pv2+g*np.maximum(0,pv2-T))-yv).mean())

# --- ensembles
m1 = qmodel(Xtr,ytr,seed=1); m2 = qmodel(Xtr,ytr,seed=2,mcw=10)
m3 = qmodel(Xtr,ytr,depth=5,mcw=15,nest=800,seed=3)
m4 = qmodel(Xtr,ytr,obj='reg:pseudohubererror',nest=600,seed=4)
m5 = qmodel(Xtr,ytr,obj='reg:squarederror',nest=600,seed=5)
P = {}
for nm,mm in [('s1',m1),('s2',m2),('d5',m3),('hub',m4),('sq',m5)]:
    P[nm] = np.clip(0.7*mm.predict(Xv)+0.3*blv,0,None)
    print(f"{nm}: MAE {np.abs(P[nm]-yv).mean():.3f}")
ens = (P['s1']+P['s2']+P['d5'])/3
print("ens(s1,s2,d5):", np.abs(ens-yv).mean())
ens2 = (P['s1']+P['s2']+P['d5']+P['hub'])/4
print("ens+hub:", np.abs(ens2-yv).mean())
ens3 = 0.5*P['s1']+0.25*P['d5']+0.25*P['sq']
print("ens q+sq:", np.abs(ens3-yv).mean())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

v = agent_api.snapshot()  # capped at day 459
tx = v.table('transactions')
print("tx shape:", tx.shape, "day range:", tx.day.min(), tx.day.max())
tx['week_no'] = ((tx.day+8)//7).astype(int)
wk = tx.groupby('week_no').agg(spend=('sales_value','sum'), hh=('household_key','nunique'))
wk['spend_per_hh'] = wk.spend/wk.hh
print("\nweekly spend per household, weeks 1..66:")
print(wk.spend_per_hh.round(1).to_string())


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.linear_model import Ridge

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0,w=None):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    m.fit(Xtr,ytr, sample_weight=w); return m

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xv,yv = make_xy(va)
blv = Xv['exp4w_blend'].values

m = qmodel(Xtr,ytr); p = m.predict(Xv)
base = np.clip(0.7*p+0.3*blv,0,None)
print("base:", np.abs(base-yv).mean())

# capping
for cap in [600,800,1000,1200,1500]:
    print(f"cap@{cap}:", np.abs(np.minimum(base,cap)-yv).mean())

# recency weighting (later snapshots weighted more)
for pw in [0.5,1.0,2.0]:
    w = (tr.snapshot_day/403)**pw
    mm = qmodel(Xtr,ytr,w=w.values); pv = mm.predict(Xv)
    f = np.clip(0.7*pv+0.3*blv,0,None)
    print(f"recency w^{pw}:", np.abs(f-yv).mean())

# hyperparams
for tag,kw in [('nest1000,lr03',dict(nest=1000,lr=0.03)),('mcw50',dict(mcw=50)),('depth3',dict(depth=3)),('col0.6',dict())]:
    if tag=='col0.6':
        mm = xgb.XGBRegressor(objective='reg:quantileerror',quantile_alpha=0.5,max_depth=4,min_child_weight=20,
                              n_estimators=600,learning_rate=0.05,subsample=0.8,colsample_bytree=0.6,n_jobs=4,random_state=0)
        mm.fit(Xtr,ytr)
    else:
        mm = qmodel(Xtr,ytr,**kw)
    pv = mm.predict(Xv); f = np.clip(0.7*pv+0.3*blv,0,None)
    print(f"hp {tag}:", np.abs(f-yv).mean())

# stacking: ridge on cal snapshot 403 (honest: model trained <=375)
tr375 = df[df.snapshot_day<=375]; cal = df[df.snapshot_day==403]
X3,y3 = make_xy(tr375); Xc,yc = make_xy(cal)
m375 = qmodel(X3,y3)
pc = m375.predict(Xc); pv2 = m375.predict(Xv)
fc = np.clip(0.7*pc+0.3*Xc['exp4w_blend'].values,0,None)
fv = np.clip(0.7*pv2+0.3*blv,0,None)
print("\nhonest stack base (<=375):", np.abs(fv-yv).mean())
S_tr = np.column_stack([fc, Xc['spend_28'], Xc['exp4w_blend'], Xc['exp4w_84'], Xc['exp4w_all'], Xc['spend_84']/3])
S_va = np.column_stack([fv, Xv['spend_28'], blv, Xv['exp4w_84'], Xv['exp4w_all'], Xv['spend_84']/3])
rg = Ridge(alpha=10.0).fit(S_tr, yc)
print("ridge stack MAE:", np.abs(rg.predict(S_va)-yv).mean(), " coefs:", rg.coef_.round(2))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings; warnings.filterwarnings('ignore')

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    m.fit(Xtr,ytr); return m

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xv,yv = make_xy(va)
blv = Xv['exp4w_blend'].values

m = qmodel(Xtr,ytr); p = m.predict(Xv)
base = np.clip(0.7*p+0.3*blv,0,None)
print("base:", round(np.abs(base-yv).mean(),3))

# (a) alpha tuning
for a in [0.45,0.5,0.52,0.55]:
    pv = qmodel(Xtr,ytr,alpha=a).predict(Xv)
    print(f"alpha={a}:", round(np.abs(np.clip(0.7*pv+0.3*blv,0,None)-yv).mean(),3))

# (b) target clipping in training
for cap in [800,1200,1500]:
    pv = qmodel(Xtr,np.minimum(ytr,cap)).predict(Xv)
    print(f"train-clip@{cap}:", round(np.abs(np.clip(0.7*pv+0.3*blv,0,None)-yv).mean(),3))

# (c) HistGradientBoosting absolute_error, then ensemble with xgb
hg = HistGradientBoostingRegressor(loss='absolute_error', max_iter=400, learning_rate=0.06,
                                   max_depth=None, min_samples_leaf=40, l2_regularization=1.0,
                                   random_state=0)
hg.fit(Xtr,ytr)
ph = hg.predict(Xv)
fh = np.clip(0.7*ph+0.3*blv,0,None)
print("HGB abs_err alone:", round(np.abs(fh-yv).mean(),3))
for w in [0.3,0.5,0.7]:
    e = w*p+(1-w)*ph
    print(f"  ens xgb({w:.1f})+hgb:", round(np.abs(np.clip(0.7*e+0.3*blv,0,None)-yv).mean(),3))

# (d) small feature subset
sub = ['exp4w_blend','spend_28','spend_84','spend_56','days_since_last','spend28_lag1','spend28_lag2',
       'trips_28','trips_84','trend28','wk_mean8','basket_mean_84','tenure','snap_day','wk_cv8','gap_mean_84']
m2 = qmodel(Xtr[sub],ytr); pv2 = m2.predict(Xv[sub])
print("subset15:", round(np.abs(np.clip(0.7*pv2+0.3*blv,0,None)-yv).mean(),3))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    m.fit(Xtr,ytr); return m

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xv,yv = make_xy(va)
blv = Xv['exp4w_blend'].values

# robustness of alpha=0.45 across seeds and depths
for seed in [0,1,2]:
    for depth,mcw in [(4,20),(4,50),(5,15)]:
        pv = qmodel(Xtr,ytr,alpha=0.45,depth=depth,mcw=mcw,seed=seed).predict(Xv)
        print(f"a=.45 d{depth} mcw{mcw} s{seed}:", round(np.abs(np.clip(0.7*pv+0.3*blv,0,None)-yv).mean(),3))

# alpha sweep finer
for a in [0.40,0.42,0.44,0.46,0.48]:
    pv = qmodel(Xtr,ytr,alpha=a).predict(Xv)
    print(f"alpha={a}:", round(np.abs(np.clip(0.7*pv+0.3*blv,0,None)-yv).mean(),3))

# blend weight with alpha=0.45
p45 = qmodel(Xtr,ytr,alpha=0.45).predict(Xv)
for w in [0.5,0.6,0.7,0.8,0.9,1.0]:
    print(f"w={w}:", round(np.abs(np.clip(w*p45+(1-w)*blv,0,None)-yv).mean(),3))

# also check on second local holdout: 403 (train<=375)
tr2 = df[df.snapshot_day<=375]; va2 = df[df.snapshot_day==403]
X2,y2 = make_xy(tr2); Xv2,yv2 = make_xy(va2); blv2 = Xv2['exp4w_blend'].values
for a in [0.45,0.5]:
    pv2 = qmodel(X2,y2,alpha=a).predict(Xv2)
    print(f"holdout403 alpha={a}:", round(np.abs(np.clip(0.7*pv2+0.3*blv2,0,None)-yv2).mean(),3))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    m.fit(Xtr,ytr); return m

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xv,yv = make_xy(va)
blv = Xv['exp4w_blend'].values
m = qmodel(Xtr,ytr); p = m.predict(Xv)
base = np.clip(0.7*p+0.3*blv,0,None)
resid = yv - base
print("resid mean:", round(resid.mean(),2), " MAE:", round(np.abs(resid).mean(),2))

# correlation of signed residual with each feature
cors = []
for c in FE:
    x = Xv[c].values.astype(float)
    ok = np.isfinite(x)
    if ok.sum()>100 and np.std(x[ok])>0:
        r = np.corrcoef(x[ok], resid[ok])[0,1]
        cors.append((c, r))
cors = sorted(cors, key=lambda t: -abs(t[1]))
print("\ntop |corr| with signed residual:")
for c,r in cors[:20]: print(f"  {c:22s} {r:+.3f}")

# reconstruction check: lag0 = spend in (s-28, s]? compare with spend_28
v = agent_api.snapshot(as_of_day=431)
tx = v.table('transactions')[['household_key','day','sales_value']]
s = 431
lag0 = tx[(tx.day>s-28)&(tx.day<=s)].groupby('household_key').sales_value.sum()
chk = feats[(feats.snapshot_day==431)].set_index('household_key')['spend_28']
cmp = pd.concat([lag0.rename('mine'), chk.rename('saved')], axis=1).dropna()
print("\nlag0 recon match:", np.abs(cmp.mine-cmp.saved).max(), " n:", len(cmp))
lag1 = tx[(tx.day>s-56)&(tx.day<=s-28)].groupby('household_key').sales_value.sum()
chk1 = feats[(feats.snapshot_day==431)].set_index('household_key')['spend28_lag1']
cmp1 = pd.concat([lag1.rename('mine'), chk1.rename('saved')], axis=1).dropna()
print("lag1 recon match:", np.abs(cmp1.mine-cmp1.saved).max(), " n:", len(cmp1))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d, cols=FE):
    X = d[cols].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0,obj='reg:quantileerror'):
    kw = dict(objective=obj, max_depth=depth, min_child_weight=mcw, n_estimators=nest,
              learning_rate=lr, subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    if obj=='reg:quantileerror': kw['quantile_alpha']=alpha
    m = xgb.XGBRegressor(**kw); m.fit(Xtr,ytr); return m

tr = df[df.snapshot_day<=403]; va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xv,yv = make_xy(va)
blv = Xv['exp4w_blend'].values
m = qmodel(Xtr,ytr); p = m.predict(Xv)
base = np.clip(0.7*p+0.3*blv,0,None)
print("base:", round(np.abs(base-yv).mean(),3))

# (a) residual boosting: target = y - exp4w_blend
mr = qmodel(Xtr, ytr - Xtr['exp4w_blend'].values)
pr = mr.predict(Xv) + blv
print("resid-boost w0.7:", round(np.abs(np.clip(0.7*pr+0.3*blv,0,None)-yv).mean(),3),
      " w1.0:", round(np.abs(np.clip(pr,0,None)-yv).mean(),3))

# (b) reg:absoluteerror
try:
    ma = qmodel(Xtr,ytr,obj='reg:absoluteerror')
    pa = ma.predict(Xv)
    print("abserr w0.7:", round(np.abs(np.clip(0.7*pa+0.3*blv,0,None)-yv).mean(),3))
    print("ens q+abserr:", round(np.abs(np.clip(0.7*(0.5*p+0.5*pa)+0.3*blv,0,None)-yv).mean(),3))
except Exception as e:
    print("abserr failed:", type(e).__name__, e)

# (c) new engineered features from existing columns
def addf(X):
    X = X.copy()
    X['overdue'] = X['days_since_last']/(X['gap_mean_84']+1)
    X['wk_max8'] = X[['wk0','wk1','wk2','wk3','wk4','wk5','wk6','wk7']].max(axis=1)
    X['wk_min8'] = X[['wk0','wk1','wk2','wk3','wk4','wk5','wk6','wk7']].min(axis=1)
    X['wk_recent4'] = X[['wk0','wk1','wk2','wk3']].mean(axis=1)
    X['wk_slope'] = (X['wk0']+X['wk1'])/2 - (X['wk6']+X['wk7'])/2
    X['lag0_x_dsl'] = X['spend_28']*np.exp(-X['days_since_last']/14)
    X['lag_ratio12'] = X['spend_28']/(X['spend28_lag1']+1)
    X['lag_ratio23'] = X['spend28_lag1']/(X['spend28_lag2']+1)
    return X

Xtr2 = addf(Xtr); Xv2 = addf(Xv)
m2 = qmodel(Xtr2,ytr); p2 = m2.predict(Xv2)
print("newfeats w0.7:", round(np.abs(np.clip(0.7*p2+0.3*blv,0,None)-yv).mean(),3))
print("newfeats w0.6:", round(np.abs(np.clip(0.6*p2+0.4*blv,0,None)-yv).mean(),3))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb
from xgboost import XGBClassifier

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    m.fit(Xtr,ytr); return m

def clf_model(Xtr,y):
    c = XGBClassifier(max_depth=4, min_child_weight=20, n_estimators=400, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, eval_metric='logloss')
    c.fit(Xtr,(y>0).astype(int)); return c

# honest: train <=375, gate-threshold tuned on 403, eval 431
X3,y3 = make_xy(df[df.snapshot_day<=375]); Xc,yc = make_xy(df[df.snapshot_day==403]); Xv,yv = make_xy(df[df.snapshot_day==431])
m375 = qmodel(X3,y3); pc = m375.predict(Xc); pv = m375.predict(Xv)
blv = Xv['exp4w_blend'].values
base = np.clip(0.7*pv+0.3*blv,0,None)
print("honest base MAE(431):", round(np.abs(base-yv).mean(),3))

cl = clf_model(X3,y3)
pzc = cl.predict_proba(Xc)[:,1]; pzv = cl.predict_proba(Xv)[:,1]
# tune threshold on 403
best = None
for t in [0.3,0.4,0.5,0.6,0.7]:
    gate = np.where(pzc>t, 0.0, 1.0)
    f = base*gate
    mae = np.abs(f-yc).mean()
    if best is None or mae<best[1]: best=(t,mae)
print("best gate threshold on 403:", best)
t = best[0]
f431 = base*np.where(pzv>t,0.0,1.0)
print("gated MAE(431):", round(np.abs(f431-yv).mean(),3))
# soft gate
for k in [0.25,0.5,1.0]:
    soft = np.clip((pzv-t)/t,0,1)  # 0 when pz>=2t? crude
    pass
soft = np.where(pzv>t, pzv*(1-pzv)*2, 1.0)  # shrink when uncertain-zero
print("soft-gated MAE(431):", round(np.abs((base*soft)-yv).mean(),3))

# also: gate + blend with exp4w_blend (blend keeps nonzero info)
f431b = np.clip(0.7*(base*np.where(pzv>t,0.0,1.0))+0.3*blv,0,None)
print("gated+blend MAE(431):", round(np.abs(f431b-yv).mean(),3))

# how many rows gated
print("frac gated at t:", round((pzv>t).mean(),3), " true zero frac:", round((yv==0).mean(),3))
# precision of gate
gz = pzv>t
print("gated rows that are truly zero:", round((yv[gz]==0).mean(),3))


# ---- cell ----

import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    m.fit(Xtr,ytr); return m

def eval_config(name, mode, w, alphas=(0.5,), seeds=(0,)):
    maes = {}
    for hd in [403, 431]:
        tr = df[df.snapshot_day < hd]; va = df[df.snapshot_day == hd]
        Xtr,ytr = make_xy(tr); Xv,yv = make_xy(va)
        blv = Xv['exp4w_blend'].values
        ps = []
        for sd in seeds:
            for a in alphas:
                if mode=='direct':
                    ps.append(qmodel(Xtr,ytr,alpha=a,seed=sd).predict(Xv))
                else:  # resid
                    ps.append(qmodel(Xtr,ytr-blv_tr_blenda if False else ytr-Xtr['exp4w_blend'].values,seed=sd).predict(Xv))
        p = np.mean(ps,axis=0)
        f = np.clip(w*p+(1-w)*blv,0,None) if mode=='direct' else np.clip(p+blv,0,None)
        maes[hd] = round(np.abs(f-yv).mean(),3)
    print(f"{name:28s} 403:{maes[403]:7.3f}  431:{maes[431]:7.3f}  avg:{np.mean(list(maes.values())):7.3f}")
    return np.mean(list(maes.values()))

eval_config("A E007 exact (w.7,s0)", 'direct', 0.7)
eval_config("B ens3 seeds w.7", 'direct', 0.7, seeds=(0,1,2))
eval_config("C ens3 seeds w.6", 'direct', 0.6, seeds=(0,1,2))
eval_config("D resid-boost ens3", 'resid', 0.0, seeds=(0,1,2))
eval_config("E ens3 alpha(.45,.5) w.7", 'direct', 0.7, alphas=(0.45,0.5), seeds=(0,1))
