import agent_api as A, pandas as pd, numpy as np
feats = A.load_saved('feats_v4.parquet')
print('feats', feats.shape)
print(feats.columns.tolist())
tt = A.train_targets()
print('targets', tt.shape)
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(count='count', mean='mean', median='median', zero_frac=lambda s:(s==0).mean())
print(g)
print(tt['future_spend_4w'].describe())
val_days=[459,487,515,543]
print('val rows in feats:', int(feats.snapshot_day.isin(val_days).sum()))
p13 = A.load_saved('pred_e013.parquet')
print('p13', p13.shape, p13.columns.tolist())
m = feats[feats.snapshot_day.isin(val_days)][['household_key','snapshot_day']].merge(p13, on=['household_key','snapshot_day'], how='left')
print('p13 merge missing:', int(m['prediction'].isna().sum()))
print('NaNs in feats:', int(feats.isna().sum().sum()))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
p13 = A.load_saved('pred_e013.parquet')
val = feats[feats.snapshot_day.isin([459,487,515,543])][['household_key','snapshot_day']]
# error analysis on TRAIN rows using in-sample? Not available. Instead analyze structure of val predictions vs train targets.
valp = val.merge(p13, on=['household_key','snapshot_day'])
print('val pred describe:'); print(valp.prediction.describe())
print('zero preds frac:', (valp.prediction<1).mean())
# train target quantiles
print('train target quantiles:', np.percentile(tt.future_spend_4w,[10,25,50,75,90,95,99]))
# how much MAE would perfect-constant-median give?
med = tt.future_spend_4w.median()
print('median-only MAE on train:', np.abs(tt.future_spend_4w-med).mean())
# distribution of train targets by spend_28 bucket (proxy for where errors live)
f = feats.merge(tt, on=['household_key','snapshot_day'])
f['b'] = pd.cut(f.spend_28, [-1,0,25,75,150,300,1e9])
print(f.groupby('b', observed=True).agg(n=('future_spend_4w','size'), mean=('future_spend_4w','mean'), med=('future_spend_4w','median'), zero=('future_spend_4w', lambda s:(s==0).mean())))
# zero-target households: what do their history features look like?
z = f[f.future_spend_4w==0]
print('zero-target n=', len(z))
print(z[['spend_28','spend_84','days_since_last','tenure']].describe().loc[['mean','50%']])
nz = f[f.future_spend_4w>0]
print(nz[['spend_28','spend_84','days_since_last','tenure']].describe().loc[['mean','50%']])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBClassifier, XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
obj_cols = [c for c in X.columns if X[c].dtype==object]
print('object cols:', obj_cols)
for c in obj_cols:
    codes = pd.factorize(X[c])[0].astype(float)
    codes[codes==-1]=np.nan
    X[c]=codes

train = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'])
y = train.future_spend_4w.values
Xi = X.values
day = feats.snapshot_day.values

def fit_reg(Xtr, ytr, alpha, depth, mcw):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, ytr); return m

mtr = day<=403; etr = day==431
Xtr, ytr = Xi[mtr], y[mtr]
Xe, ye = Xi[etr], y[etr]
t0=time.time()
preds=[]
for d,mcw in [(4,20),(5,40),(6,60)]:
    m=fit_reg(Xtr,ytr,0.5,d,mcw); preds.append(m.predict(Xe))
base=np.clip(np.mean(preds),0,None)
print('baseline q50(all) MAE on 431:', round(np.abs(base-ye).mean(),3), 'time', round(time.time()-t0,1))

t0=time.time()
clf = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, tree_method='hist',
                    eval_metric='logloss')
clf.fit(Xtr, (ytr==0).astype(int))
p0 = clf.predict_proba(Xe)[:,1]
pos = ytr>0
qgrid=[0.30,0.40,0.50,0.60,0.70]
qpreds={}
for a in qgrid:
    ps=[]
    for d,mcw in [(4,20),(5,40),(6,60)]:
        m=fit_reg(Xtr[pos],ytr[pos],a,d,mcw); ps.append(m.predict(Xe))
    qpreds[a]=np.clip(np.mean(ps),0,None)
Q=np.column_stack([qpreds[a] for a in qgrid])
qn=np.clip((0.5-p0)/(1-p0), qgrid[0], qgrid[-1])
two=np.zeros(len(ye))
zero_mask = p0>=0.5
idx=np.searchsorted(qgrid, qn)
lo=np.clip(idx-1,0,len(qgrid)-1); hi=np.clip(idx,0,len(qgrid)-1)
w=np.where(qgrid[hi]>qgrid[lo],(qn-qgrid[lo])/np.maximum(qgrid[hi]-qgrid[lo],1e-9),0)
two[~zero_mask]=Q[np.arange(len(ye)),lo]*(1-w[~zero_mask])+Q[np.arange(len(ye)),hi]*w[~zero_mask]
print('two-part mixture-median MAE on 431:', round(np.abs(two-ye).mean(),3), 'time', round(time.time()-t0,1))
print('actual zero frac:', round((ye==0).mean(),3), '| two-part pred-zero frac:', round(zero_mask.mean(),3), '| baseline pred<1 frac:', round((base<1).mean(),3))
# also: plain q50 on positives-only with p0 rule at fixed 0.5 (no interp) for ablation
twoB = np.where(zero_mask, 0.0, qpreds[0.50])
print('two-part fixed-q50 MAE on 431:', round(np.abs(twoB-ye).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBClassifier, XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if X[c].dtype==object:
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
print('aligned:', len(y_full), 'nan(val):', int(np.isnan(y_full).sum()))

def fit_reg(Xtr, ytr, alpha, depth, mcw):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, ytr); return m

mtr = (day<=403) & ~np.isnan(y_full); etr = (day==431)
Xtr, ytr = Xi[mtr], y_full[mtr]
Xe, ye = Xi[etr], y_full[etr]
t0=time.time()
preds=[]
for d,mcw in [(4,20),(5,40),(6,60)]:
    m=fit_reg(Xtr,ytr,0.5,d,mcw); preds.append(m.predict(Xe))
base=np.clip(np.mean(preds),0,None)
print('baseline q50(all) MAE on 431:', round(np.abs(base-ye).mean(),3), 'time', round(time.time()-t0,1))

t0=time.time()
clf = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, tree_method='hist',
                    eval_metric='logloss')
clf.fit(Xtr, (ytr==0).astype(int))
p0 = clf.predict_proba(Xe)[:,1]
pos = ytr>0
qgrid=[0.30,0.40,0.50,0.60,0.70]
qpreds={}
for a in qgrid:
    ps=[]
    for d,mcw in [(4,20),(5,40),(6,60)]:
        m=fit_reg(Xtr[pos],ytr[pos],a,d,mcw); ps.append(m.predict(Xe))
    qpreds[a]=np.clip(np.mean(ps),0,None)
Q=np.column_stack([qpreds[a] for a in qgrid])
qn=np.clip((0.5-p0)/(1-p0), qgrid[0], qgrid[-1])
zero_mask = p0>=0.5
idx=np.clip(np.searchsorted(qgrid, qn),0,len(qgrid)-1)
lo=np.clip(idx-1,0,len(qgrid)-1); hi=idx
w=np.where(qgrid[hi]>qgrid[lo],(qn-qgrid[lo])/np.maximum(qgrid[hi]-qgrid[lo],1e-9),0)
two=Q[np.arange(len(ye)),lo]*(1-w)+Q[np.arange(len(ye)),hi]*w
two[zero_mask]=0.0
print('two-part mixture-median MAE on 431:', round(np.abs(two-ye).mean(),3), 'time', round(time.time()-t0,1))
print('actual zero frac:', round((ye==0).mean(),3), '| two-part pred-zero frac:', round(zero_mask.mean(),3), '| baseline pred<1 frac:', round((base<1).mean(),3))
twoB = np.where(zero_mask, 0.0, qpreds[0.50])
print('two-part fixed-q50 MAE on 431:', round(np.abs(twoB-ye).mean(),3))
# single-model comparisons
m1=fit_reg(Xtr,ytr,0.5,4,20); print('single q50 d4 MAE:', round(np.abs(np.clip(m1.predict(Xe),0,None)-ye).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBClassifier, XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
print(X.dtypes.value_counts())
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values

def fit_reg(Xtr, ytr, alpha, depth, mcw):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, ytr); return m

mtr = (day<=403) & ~np.isnan(y_full); etr = (day==431)
Xtr, ytr = Xi[mtr], y_full[mtr]
Xe, ye = Xi[etr], y_full[etr]
t0=time.time()
preds=[]
for d,mcw in [(4,20),(5,40),(6,60)]:
    m=fit_reg(Xtr,ytr,0.5,d,mcw); preds.append(m.predict(Xe))
base=np.clip(np.mean(preds),0,None)
print('baseline q50(all) MAE on 431:', round(np.abs(base-ye).mean(),3), 'time', round(time.time()-t0,1))

t0=time.time()
clf = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, tree_method='hist',
                    eval_metric='logloss')
clf.fit(Xtr, (ytr==0).astype(int))
p0 = clf.predict_proba(Xe)[:,1]
pos = ytr>0
qgrid=[0.30,0.40,0.50,0.60,0.70]
qpreds={}
for a in qgrid:
    ps=[]
    for d,mcw in [(4,20),(5,40),(6,60)]:
        m=fit_reg(Xtr[pos],ytr[pos],a,d,mcw); ps.append(m.predict(Xe))
    qpreds[a]=np.clip(np.mean(ps),0,None)
Q=np.column_stack([qpreds[a] for a in qgrid])
qn=np.clip((0.5-p0)/(1-p0), qgrid[0], qgrid[-1])
zero_mask = p0>=0.5
idx=np.clip(np.searchsorted(qgrid, qn),0,len(qgrid)-1)
lo=np.clip(idx-1,0,len(qgrid)-1); hi=idx
w=np.where(qgrid[hi]>qgrid[lo],(qn-qgrid[lo])/np.maximum(qgrid[hi]-qgrid[lo],1e-9),0)
two=Q[np.arange(len(ye)),lo]*(1-w)+Q[np.arange(len(ye)),hi]*w
two[zero_mask]=0.0
print('two-part mixture-median MAE on 431:', round(np.abs(two-ye).mean(),3), 'time', round(time.time()-t0,1))
print('actual zero frac:', round((ye==0).mean(),3), '| two-part pred-zero frac:', round(zero_mask.mean(),3), '| baseline pred<1 frac:', round((base<1).mean(),3))
twoB = np.where(zero_mask, 0.0, qpreds[0.50])
print('two-part fixed-q50 MAE on 431:', round(np.abs(twoB-ye).mean(),3))
m1=fit_reg(Xtr,ytr,0.5,4,20); print('single q50 d4 MAE:', round(np.abs(np.clip(m1.predict(Xe),0,None)-ye).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBClassifier, XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values

def fit_reg(Xtr, ytr, alpha, depth, mcw):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, ytr); return m

mtr = (day<=403) & ~np.isnan(y_full); etr = (day==431)
Xtr, ytr = Xi[mtr], y_full[mtr]
Xe, ye = Xi[etr], y_full[etr]
t0=time.time()
preds=[]
for d,mcw in [(4,20),(5,40),(6,60)]:
    m=fit_reg(Xtr,ytr,0.5,d,mcw); preds.append(m.predict(Xe))
base=np.clip(np.mean(preds),0,None)
print('baseline q50(all) MAE on 431:', round(np.abs(base-ye).mean(),3), 'time', round(time.time()-t0,1))

t0=time.time()
clf = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, tree_method='hist',
                    eval_metric='logloss')
clf.fit(Xtr, (ytr==0).astype(int))
p0 = clf.predict_proba(Xe)[:,1]
pos = ytr>0
qgrid=[0.30,0.40,0.50,0.60,0.70]
qpreds={}
for a in qgrid:
    ps=[]
    for d,mcw in [(4,20),(5,40),(6,60)]:
        m=fit_reg(Xtr[pos],ytr[pos],a,d,mcw); ps.append(m.predict(Xe))
    qpreds[a]=np.clip(np.mean(ps),0,None)
Q=np.column_stack([qpreds[a] for a in qgrid])
qn=np.clip((0.5-p0)/(1-p0), qgrid[0], qgrid[-1])
zero_mask = p0>=0.5
idx=np.clip(np.searchsorted(qgrid, qn),0,len(qgrid)-1)
lo=np.clip(idx-1,0,len(qgrid)-1); hi=idx
w=np.where(qgrid[hi]>qgrid[lo],(qn-qgrid[lo])/np.maximum(qgrid[hi]-qgrid[lo],1e-9),0)
two=Q[np.arange(len(ye)),lo]*(1-w)+Q[np.arange(len(ye)),hi]*w
two[zero_mask]=0.0
print('two-part mixture-median MAE on 431:', round(np.abs(two-ye).mean(),3), 'time', round(time.time()-t0,1))
print('actual zero frac:', round((ye==0).mean(),3), '| two-part pred-zero frac:', round(zero_mask.mean(),3), '| baseline pred<1 frac:', round((base<1).mean(),3))
twoB = np.where(zero_mask, 0.0, qpreds[0.50])
print('two-part fixed-q50 MAE on 431:', round(np.abs(twoB-ye).mean(),3))
m1=fit_reg(Xtr,ytr,0.5,4,20); print('single q50 d4 MAE:', round(np.abs(np.clip(m1.predict(Xe),0,None)-ye).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
etr = day==431
ye = y_full[etr]
print('day431 n:', ye.sum()>0, len(ye), 'mean', round(np.nanmean(ye),1), 'median', round(np.nanmedian(ye),1), 'zero frac', round((ye==0).mean(),3))
print('constant median MAE on 431:', round(np.abs(ye-np.nanmedian(y_full[day<=403])).mean(),3))
for f in ['exp4w_blend','spend_28','spend_84','lag1_spend']:
    v = feats[f].values[etr]
    print(f, 'MAE on 431:', round(np.abs(np.nan_to_num(v)-ye).mean(),3), 'corr:', round(np.corrcoef(np.nan_to_num(v), ye)[0,1],3))
# quick model again with more care
mtr = (day<=403) & ~np.isnan(y_full)
Xtr, ytr = Xi[mtr], y_full[mtr]
m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                 subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
                 n_jobs=4, random_state=0, tree_method='hist')
t0=time.time(); m.fit(Xtr, ytr); pr = np.clip(m.predict(Xi[etr]),0,None)
print('q50 d4 MAE on 431:', round(np.abs(pr-ye).mean(),3), 'time', round(time.time()-t0,1))
print('pred stats:', np.round(np.percentile(pr,[10,50,90]),1), 'target stats:', np.round(np.percentile(ye,[10,50,90]),1))
# error by target bucket
b = pd.cut(ye, [-1,0,25,75,150,300,1e9])
df = pd.DataFrame({'ye':ye,'pr':pr,'b':b})
print(df.groupby('b', observed=True).agg(n=('ye','size'), med_t=('ye','median'), med_p=('pr','median'), mae=('ye', lambda s: np.abs(s-df.loc[s.index,'pr']).mean())))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBClassifier, XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
etr = day==431; ye = y_full[etr]
mtr = (day<=403) & ~np.isnan(y_full)
Xtr, ytr = Xi[mtr], y_full[mtr]
Xe = Xi[etr]

def fit_reg(Xtr, ytr, alpha, depth, mcw):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, ytr); return m

t0=time.time()
clf = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, tree_method='hist',
                    eval_metric='logloss')
clf.fit(Xtr, (ytr==0).astype(int))
p0 = clf.predict_proba(Xe)[:,1]
pos = ytr>0
qgrid=[0.30,0.40,0.50,0.60,0.70]
qpreds={}
for a in qgrid:
    ps=[]
    for d,mcw in [(4,20),(5,40),(6,60)]:
        m=fit_reg(Xtr[pos],ytr[pos],a,d,mcw); ps.append(m.predict(Xe))
    qpreds[a]=np.clip(np.mean(ps),0,None)
Q=np.column_stack([qpreds[a] for a in qgrid])
qn=np.clip((0.5-p0)/(1-p0), qgrid[0], qgrid[-1])
zero_mask = p0>=0.5
idx=np.clip(np.searchsorted(qgrid, qn),0,len(qgrid)-1)
lo=np.clip(idx-1,0,len(qgrid)-1); hi=idx
w=np.where(qgrid[hi]>qgrid[lo],(qn-qgrid[lo])/np.maximum(qgrid[hi]-qgrid[lo],1e-9),0)
two=Q[np.arange(len(ye)),lo]*(1-w)+Q[np.arange(len(ye)),hi]*w
two[zero_mask]=0.0
print('two-part mixture-median MAE on 431:', round(np.abs(two-ye).mean(),3), 'time', round(time.time()-t0,1))
print('actual zero frac:', round((ye==0).mean(),3), '| two-part pred-zero frac:', round(zero_mask.mean(),3))
twoB = np.where(zero_mask, 0.0, qpreds[0.50])
print('two-part fixed-q50 MAE on 431:', round(np.abs(twoB-ye).mean(),3))
# also check p0 calibration
from sklearn.metrics import brier_score_loss
print('brier p0:', round(brier_score_loss((ye==0).astype(int), p0),4))
b = pd.cut(ye, [-1,0,25,75,150,300,1e9])
df = pd.DataFrame({'ye':ye,'two':two,'p0':p0,'b':b})
print(df.groupby('b', observed=True).agg(n=('ye','size'), med_t=('ye','median'), med_p=('two','median'), mae=('ye', lambda s: np.abs(s-df.loc[s.index,'two']).mean()), meanp0=('p0','mean')))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import brier_score_loss

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
etr = day==431; ye = y_full[etr]
mtr = (day<=403) & ~np.isnan(y_full)
Xtr, ytr = Xi[mtr], y_full[mtr]
Xe = Xi[etr]

def fit_reg(Xtr, ytr, alpha, depth, mcw):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, ytr); return m

t0=time.time()
clf = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, tree_method='hist',
                    eval_metric='logloss')
clf.fit(Xtr, (ytr==0).astype(int))
p0 = clf.predict_proba(Xe)[:,1]
pos = ytr>0
qgrid=np.array([0.30,0.40,0.50,0.60,0.70])
qpreds={}
for a in qgrid:
    ps=[]
    for d,mcw in [(4,20),(5,40),(6,60)]:
        m=fit_reg(Xtr[pos],ytr[pos],a,d,mcw); ps.append(m.predict(Xe))
    qpreds[a]=np.clip(np.mean(ps,0),0,None)
Q=np.column_stack([qpreds[a] for a in qgrid])
qn=np.clip((0.5-p0)/(1-p0), qgrid[0], qgrid[-1])
zero_mask = p0>=0.5
idx=np.clip(np.searchsorted(qgrid, qn),0,len(qgrid)-1)
lo=np.clip(idx-1,0,len(qgrid)-1); hi=idx
w=np.where(qgrid[hi]>qgrid[lo],(qn-qgrid[lo])/np.maximum(qgrid[hi]-qgrid[lo],1e-9),0)
two=Q[np.arange(len(ye)),lo]*(1-w)+Q[np.arange(len(ye)),hi]*w
two[zero_mask]=0.0
print('two-part mixture-median MAE on 431:', round(np.abs(two-ye).mean(),3), 'time', round(time.time()-t0,1))
print('actual zero frac:', round((ye==0).mean(),3), '| two-part pred-zero frac:', round(zero_mask.mean(),3), '| brier:', round(brier_score_loss((ye==0).astype(int), p0),4))
twoB = np.where(zero_mask, 0.0, qpreds[0.50])
print('two-part fixed-q50 MAE on 431:', round(np.abs(twoB-ye).mean(),3))
# ablation: quantile-mapping without p0 (i.e., p0=0): qn=0.5 -> q50
twoC = qpreds[0.50]
print('positives-only q50 (no zero rule) MAE:', round(np.abs(twoC-ye).mean(),3))
b = pd.cut(ye, [-1,0,25,75,150,300,1e9])
df = pd.DataFrame({'ye':ye,'two':two,'p0':p0,'b':b})
print(df.groupby('b', observed=True).agg(n=('ye','size'), med_t=('ye','median'), med_p=('two','median'), mae=('ye', lambda s: np.abs(s-df.loc[s.index,'two']).mean()), meanp0=('p0','mean')))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBClassifier, XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
etr = day==431; ye = y_full[etr]
mtr = (day<=403) & ~np.isnan(y_full)
Xtr, ytr = Xi[mtr], y_full[mtr]
Xe = Xi[etr]

def fit_reg(Xtr, ytr, alpha, depth, mcw):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, ytr); return m

clf = XGBClassifier(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=0, tree_method='hist',
                    eval_metric='logloss')
clf.fit(Xtr, (ytr==0).astype(int))
p0 = clf.predict_proba(Xe)[:,1]
pos = ytr>0
qgrid=np.array([0.30,0.40,0.50,0.60,0.70])
qpreds={}
for a in qgrid:
    ps=[]
    for d,mcw in [(4,20),(5,40),(6,60)]:
        m=fit_reg(Xtr[pos],ytr[pos],a,d,mcw); ps.append(m.predict(Xe))
    qpreds[a]=np.clip(np.mean(ps,0),0,None)
# proper baseline: q50 on ALL rows, 3-model elementwise mean
ps_all=[]
for d,mcw in [(4,20),(5,40),(6,60)]:
    m=fit_reg(Xtr,ytr,0.5,d,mcw); ps_all.append(m.predict(Xe))
base=np.clip(np.mean(ps_all,0),0,None)
print('baseline 3-model q50(all) MAE on 431:', round(np.abs(base-ye).mean(),3))

Q=np.column_stack([qpreds[a] for a in qgrid])
qn=np.clip((0.5-p0)/(1-p0), qgrid[0], qgrid[-1])
zero_mask = p0>=0.5
idx=np.clip(np.searchsorted(qgrid, qn),0,len(qgrid)-1)
lo=np.clip(idx-1,0,len(qgrid)-1); hi=idx
w=np.where(qgrid[hi]>qgrid[lo],(qn-qgrid[lo])/np.maximum(qgrid[hi]-qgrid[lo],1e-9),0)
two=Q[np.arange(len(ye)),lo]*(1-w)+Q[np.arange(len(ye)),hi]*w
two[zero_mask]=0.0
print('two-part mixture-median MAE:', round(np.abs(two-ye).mean(),3))
# variant A: no hard zero, pure mixture scaling: (1-p0)*q_mapped
mixA=(1-p0)*Q[np.arange(len(ye)),lo]*(1-w)+ (1-p0)*Q[np.arange(len(ye)),hi]*w
print('variant A (1-p0)*q_mapped MAE:', round(np.abs(mixA-ye).mean(),3))
# variant B: two-part but threshold sweep
for thr in [0.4,0.45,0.5,0.55,0.6]:
    t=np.where(p0>=thr, 0.0, two)
    print(f'two-part thr={thr} MAE:', round(np.abs(t-ye).mean(),3))
# variant C: two-part with qn = 0.5 - 0.5*p0 (gentler mapping)
qn2=np.clip(0.5-0.5*p0, qgrid[0], qgrid[-1])
idx2=np.clip(np.searchsorted(qgrid, qn2),0,len(qgrid)-1)
lo2=np.clip(idx2-1,0,len(qgrid)-1); hi2=idx2
w2=np.where(qgrid[hi2]>qgrid[lo2],(qn2-qgrid[lo2])/np.maximum(qgrid[hi2]-qgrid[lo2],1e-9),0)
twoC=Q[np.arange(len(ye)),lo2]*(1-w2)+Q[np.arange(len(ye)),hi2]*w2
twoC=np.where(zero_mask,0.0,twoC)
print('variant C (qn=0.5-0.5p0) MAE:', round(np.abs(twoC-ye).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
etr = day==431; ye = y_full[etr]
mtr = (day<=403) & ~np.isnan(y_full)
Xtr, ytr = Xi[mtr], y_full[mtr]
Xe = Xi[etr]

configs = [(4,20),(4,40),(5,20),(5,40),(5,60),(6,40),(6,60),(4,60)]
def fit(alpha, depth, mcw, target):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, target); return m

t0=time.time()
raw=[]; lg=[]
for d,mcw in configs:
    raw.append(fit(0.5,d,mcw,ytr).predict(Xe))
    lg.append(np.expm1(fit(0.5,d,mcw,np.log1p(ytr)).predict(Xe)))
raw=np.clip(np.mean(np.array(raw),0),0,None)
lg=np.clip(np.mean(np.array(lg),0),0,None)
print('raw-scale 8-bag MAE on 431:', round(np.abs(raw-ye).mean(),3))
print('log-scale 8-bag MAE on 431:', round(np.abs(lg-ye).mean(),3), 'time', round(time.time()-t0,1))
for w in [0.25,0.5,0.75]:
    b=w*lg+(1-w)*raw
    print(f'blend w={w} log MAE:', round(np.abs(b-ye).mean(),3))
b=pd.cut(ye,[-1,0,25,75,150,300,1e9])
df=pd.DataFrame({'ye':ye,'raw':raw,'lg':lg,'b':b})
print(df.groupby('b',observed=True).agg(n=('ye','size'), med_t=('ye','median'), med_raw=('raw','median'), med_log=('lg','median')))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
etr = day==431; ye = y_full[etr]
blend_feat = np.nan_to_num(feats['exp4w_blend'].values)

configs = [(4,20),(4,40),(5,20),(5,40),(5,60),(6,40),(6,60),(4,60)]
def fit(Xtr, ytr, depth, mcw, seed):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
                     n_jobs=4, random_state=seed, tree_method='hist')
    m.fit(Xtr, ytr); return m

t0=time.time()
# local check on 431: train <=403, 8 configs x 2 seeds
mtr = (day<=403) & ~np.isnan(y_full)
preds=[]
for d,mcw in configs:
    for s in [0,1]:
        preds.append(fit(Xi[mtr], y_full[mtr], d, mcw, s).predict(Xi[etr]))
bag16 = np.clip(np.mean(np.array(preds),0),0,None)
print('8cfg x 2seed MAE on 431:', round(np.abs(bag16-ye).mean(),3), 'time', round(time.time()-t0,1))
for w in [0.6,0.7,0.8,1.0]:
    f = np.clip(w*bag16 + (1-w)*blend_feat[etr], 0, None)
    print(f'blend model*{w}+exp4w*{round(1-w,1)} MAE:', round(np.abs(f-ye).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
val_mask = day>=459
train_mask = (day<=431) & ~np.isnan(y_full)
Xtr, ytr = Xi[train_mask], y_full[train_mask]
Xval = Xi[val_mask]
print('train rows:', len(ytr), 'val rows:', int(val_mask.sum()))

configs = [(4,20),(4,40),(5,20),(5,40),(5,60),(6,40),(6,60),(4,60)]
t0=time.time()
preds=[]
for d,mcw in configs:
    for s in [0,1]:
        m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=d, min_child_weight=mcw,
                         subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
                         n_jobs=4, random_state=s, tree_method='hist')
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xval))
pred = np.clip(np.mean(np.array(preds),0), 0, None)
out = feats.loc[val_mask, ['household_key','snapshot_day']].copy()
out['prediction'] = pred
print('out', out.shape, 'nan:', int(out.prediction.isna().sum()), 'finite:', bool(np.isfinite(out.prediction).all()))
p = A.save_table(out, 'pred_e014.parquet')
print('saved:', p, 'time', round(time.time()-t0,1))
