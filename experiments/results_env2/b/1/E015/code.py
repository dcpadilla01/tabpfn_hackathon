
import agent_api as A, pandas as pd, numpy as np
for name in ['e008_level_shape','e006_cadence','e007_temporal','e001_history','e003_full']:
    df = A.load_saved(name + '.parquet')
    print('==', name, df.shape)
    print(list(df.columns))
tt = A.train_targets()
print('targets', tt.shape)
print(tt['future_spend_4w'].describe())
v = A.snapshot()
tx = v.table('transactions')
print(tx.dtypes)
print('neg sales', int((tx.sales_value<0).sum()), 'qty<=0', int((tx.quantity<=0).sum()))
p = v.table('products')
print(p['brand'].value_counts(dropna=False).head(10))
print('n commodities', p['commodity_desc'].nunique(), 'n depts', p['department'].nunique())
print('households type:', type(v.households))
print(list(pd.Index(v.households))[:3] if not isinstance(v.households, pd.DataFrame) else v.households.head(3))


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
e8 = A.load_saved('e008_level_shape.parquet')
tt = A.train_targets()
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
def mae(p): return float(np.mean(np.abs(np.asarray(p)-y)))
print('pred0', round(mae(np.zeros(len(y))),2), 'const-median', round(mae(np.full(len(y), np.median(y))),2))
for c in ['sp28','sp84','sp364','sp728','z_med4w_hist','z_max4w_hist','sp28_rate','sp364_rate','splag1y','sp_lag1y','newma28','wksp_mean','tenure','days_since_last']:
    if c in m: print(c, round(mae(m[c].fillna(0).values),2))
print('blend sp28/med4w', round(mae(0.5*m['sp28'].fillna(0)+0.5*m['z_med4w_hist'].fillna(0)),2))
print('blend sp28/84/med', round(mae((m['sp28'].fillna(0)+m['sp84'].fillna(0)+m['z_med4w_hist'].fillna(0))/3),2))
# rank corr of all numeric features with target
num = [c for c in e8.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
ranks = m[num].rank()
yc = pd.Series(y).rank()
cor = {}
for c in num:
    s = ranks[c]; ok = s.notna()
    if ok.sum()>500: cor[c] = abs(float(np.corrcoef(s[ok], yc[ok])[0,1]))
top = sorted(cor.items(), key=lambda kv:-kv[1])
print('TOP30:', [(k, round(v,3)) for k,v in top[:30]])
print('BOT10:', [(k, round(v,3)) for k,v in top[-10:]])
print('per-snapshot target mean/median:')
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']))


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
e8 = A.load_saved('e008_level_shape.parquet')
tt = A.train_targets()
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
# Quantile-bucket analysis: how well does sp28 (best single) predict in each bucket?
q = pd.qcut(m['sp28'].fillna(0), 10, duplicates='drop')
g = m.groupby(q, observed=True).agg(y=('future_spend_4w','mean'), pred=('sp28','mean'), n=('future_spend_4w','size'))
g['y_med'] = m.groupby(q, observed=True)['future_spend_4w'].median()
print(g)
# zero rate
print('zero rate', (y==0).mean())
# key: does a log-target view help? check corr of log1p(y) with log1p(sp28)
print('corr log', np.corrcoef(np.log1p(m['sp28'].fillna(0)), np.log1p(y))[0,1])
# what does sp28 look like vs y in raw scale
print(m[['sp28','future_spend_4w']].describe())
# check per-snapshot: is val harder?
print('train sp28 mae by snapshot:')
m['mae_sp28'] = np.abs(m['sp28'].fillna(0)-y)
print(m.groupby('snapshot_day')['mae_sp28'].mean())
# distribution of sp28 vs y at each snapshot - drift?
for d in [95, 431, 459, 543]:
    pass
# look at raw data around snapshots: any structural break in spend level over time?
v = A.snapshot()
tx = v.table('transactions')
tx = tx[tx.day<=459]
wk = tx.groupby('day')['sales_value'].sum()
print('total spend by 28d period:')
print((wk.groupby((wk.index-1)//28).sum()).round(0).to_string())


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
from collections import defaultdict
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
hk = m['household_key'].values; sd = m['snapshot_day'].values
med = np.median(y)
def mae(p): return float(np.mean(np.abs(np.asarray(p,float)-y)))
ymap = {(h,s):v for h,s,v in zip(tt.household_key,tt.snapshot_day,tt.future_spend_4w)}
ar1 = np.array([ymap.get((h,s-28),np.nan) for h,s in zip(hk,sd)])
ar2 = np.array([ymap.get((h,s-56),np.nan) for h,s in zip(hk,sd)])
ok1=~np.isnan(ar1); ok2=~np.isnan(ar2)
print('AR1 cov',round(ok1.mean(),3),'MAE fill-med',round(mae(np.where(ok1,ar1,med)),2))
print('AR1+2 blend', round(mae(np.where(ok1&ok2,0.5*ar1+0.5*ar2,np.where(ok1,ar1,med))),2))
byh=defaultdict(list)
for h,s,v in zip(tt.household_key,tt.snapshot_day,tt.future_spend_4w): byh[h].append((s,v))
tsmean=np.full(len(y),np.nan)
for i,(h,s) in enumerate(zip(hk,sd)):
    past=[v for s2,v in byh[h] if s2<s]
    if past: tsmean[i]=np.mean(past)
okt=~np.isnan(tsmean)
print('TSmean cov',round(okt.mean(),3),'MAE',round(mae(np.where(okt,tsmean,med)),2))
print('blend .5sp28+.5tsmean',round(mae(0.5*m['sp28'].fillna(0)+0.5*np.where(okt,tsmean,med)),2))
print('blend .6/.4',round(mae(0.6*m['sp28'].fillna(0)+0.4*np.where(okt,tsmean,med)),2))
print('oracle hh-mean insample',round(mae(m.groupby('household_key')['future_spend_4w'].transform('mean')),2))
print('sp28 vs ar1 corr',round(np.corrcoef(m['sp28'].fillna(0)[ok1],ar1[ok1])[0,1],4),'maxdiff',round(np.abs(m['sp28'].fillna(0)[ok1]-ar1[ok1]).max(),2))
# ridge proxy CV: fit sd<=347, eval 375/403/431
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
X=m[feats].astype(float); X=X.fillna(X.median()).values
mu=X.mean(0); sg=X.std(0)+1e-9; Xz=(X-mu)/sg
def ridge(lam,tr,ev):
    Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
    w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(feats)), Xt.T@(yt-ym))
    return float(np.mean(np.abs(Xz[ev]@w+ym-y[ev])))
tr=sd<=347; ev=(sd>=375)
for lam in [0.3,1,3,10,30,100]:
    print('ridge lam',lam,'MAE 375-431',round(ridge(lam,tr,ev),2))
# splag1y inspection
for c in ['splag1y','sp_lag1y']:
    s=m[c]; ok=s.notna()
    print(c,'cov',round(ok.mean(),3),'corr',round(np.corrcoef(s[ok],y[ok])[0,1],3),'mean',round(s[ok].mean(),1))
print('splag1y cov by snap:',m.groupby('snapshot_day')['splag1y'].apply(lambda x:round(x.notna().mean(),2)).to_string())
# e009 leakage check
e9=A.load_saved('e009_target_enc.parquet')
extra=[c for c in e9.columns if c not in e8.columns]
print('e009 extra cols:',extra)
m9=tt.merge(e9,on=['household_key','snapshot_day'],how='left')
for c in extra:
    s=m9[c].astype(float); ok=s.notna()
    print(c,'cov',round(ok.mean(),3),'corrY',round(np.corrcoef(s[ok],m9['future_spend_4w'].values[ok])[0,1],3))


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
from collections import defaultdict
tt = A.train_targets()
v = A.snapshot(459)
txg = v.table('transactions').groupby(['household_key','day'])['sales_value'].sum().reset_index()
ser = {h: g.set_index('day').sales_value for h,g in txg.groupby('household_key')}
def lag_fast(h,s,back):
    ss = ser.get(h)
    if ss is None: return 0.0
    idx = ss.index
    lo, hi = s+1-back, s+28-back
    sel = idx[(idx>=lo)&(idx<=hi)]
    return float(ss.loc[sel].sum()) if len(sel) else 0.0
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
hk = m['household_key'].values; sd = m['snapshot_day'].values
lag1y = np.array([lag_fast(h,s,364) for h,s in zip(hk,sd)])
lag2y = np.array([lag_fast(h,s,728) for h,s in zip(hk,sd)])
lag13 = np.array([lag_fast(h,s,336) for h,s in zip(hk,sd)])
med = np.median(y)
def mae(p): return float(np.mean(np.abs(np.asarray(p,float)-y)))
for nm,arr in [('lag1y',lag1y),('lag2y',lag2y),('lag13',lag13)]:
    print(nm,'mean',round(arr.mean(),1),'corr',round(np.corrcoef(arr,y)[0,1],3),'MAE',round(mae(arr),2))
print('blend sp28+lag1y', round(mae(0.5*m['sp28'].fillna(0)+0.5*lag1y),2))
print('blend sp28+.3lag1y', round(mae(0.7*m['sp28'].fillna(0)+0.3*lag1y),2))
byh=defaultdict(list)
for h,s,vv in zip(tt.household_key,tt.snapshot_day,tt.future_spend_4w): byh[h].append((s,vv))
tsmean=np.full(len(y),np.nan)
for i,(h,s) in enumerate(zip(hk,sd)):
    past=[vv for s2,vv in byh[h] if s2<s]
    if past: tsmean[i]=np.mean(past)
okt=~np.isnan(tsmean)
print('tsmean MAE',round(mae(np.where(okt,tsmean,med)),2))
print('blend sp28+tsmean',round(mae(0.5*m['sp28'].fillna(0)+0.5*np.where(okt,tsmean,med)),2))
print('blend sp28+tsmean+lag1y',round(mae((m['sp28'].fillna(0)+np.where(okt,tsmean,med)+lag1y)/3),2))
print('ridge proxy check on lag1y alone: corr', round(np.corrcoef(np.log1p(lag1y),np.log1p(y))[0,1],3))


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
from collections import defaultdict
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
hk = m['household_key'].values; sd = m['snapshot_day'].values
# seasonal lags from transactions
v = A.snapshot(459)
txg = v.table('transactions').groupby(['household_key','day'])['sales_value'].sum().reset_index()
ser = {h: g.set_index('day').sales_value for h,g in txg.groupby('household_key')}
def lag_fast(h,s,back):
    ss = ser.get(h)
    if ss is None: return 0.0
    idx = ss.index; lo, hi = s+1-back, s+28-back
    sel = idx[(idx>=lo)&(idx<=hi)]
    return float(ss.loc[sel].sum()) if len(sel) else 0.0
lag13 = np.array([lag_fast(h,s,336) for h,s in zip(hk,sd)])
lag1y = np.array([lag_fast(h,s,364) for h,s in zip(hk,sd)])
lag15 = np.array([lag_fast(h,s,392) for h,s in zip(hk,sd)])
# proxy ridge
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
def prep(Xdf, cols):
    X=Xdf[cols].astype(float); X=X.fillna(X.median()); 
    mu=X.mean(0); sg=X.std(0)+1e-9; return (X.values-mu.values)/sg.values, mu.values, sg.values
def run(cols, lams=(30,100,300)):
    Xz,mu,sg = prep(m, cols)
    tr=sd<=347; ev=sd>=375
    out={}
    for lam in lams:
        Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
        w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(cols)), Xt.T@(yt-ym))
        out[lam]=round(float(np.mean(np.abs(Xz[ev]@w+ym-y[ev]))),2)
    return out
print('base E008:', run(feats))
# + seasonal
m2 = m.copy()
m2['slag13']=lag13; m2['slag1y']=lag1y; m2['slag15']=lag15
m2['slag13_r'] = lag13/(m2['sp84'].fillna(0)+1)
print('+seasonal:', run(feats+['slag13','slag1y','slag15','slag13_r']))
# interactions of top features
m2['i_sp28_trend'] = m2['sp28'].fillna(0)*m2['trend_84'].fillna(0)
m2['i_sp28_nact'] = m2['sp28'].fillna(0)*m2['nact28'].fillna(0)
m2['i_sp28_demo4'] = m2['sp28'].fillna(0)*(m2['d_classification_4'].fillna('0')=='5+').astype(float)
m2['i_sp28_kid'] = m2['sp28'].fillna(0)*(m2['d_kid_category_desc'].fillna('None/Unknown').isin(['1','2','3+'])).astype(float)
m2['i_newma_rec'] = m2['newma28'].fillna(0)*m2['days_since_last'].fillna(0)
m2['sq_sp28'] = m2['sp28'].fillna(0)**2
m2['sp28_x_sp84rate'] = m2['sp28'].fillna(0)*m2['sp84_rate'].fillna(0)
print('+interact:', run(feats+['i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))
print('+both:', run(feats+['slag13','slag1y','slag15','slag13_r','i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
hk = m['household_key'].values; sd = m['snapshot_day'].values
v = A.snapshot(459)
txg = v.table('transactions').groupby(['household_key','day'])['sales_value'].sum().reset_index()
ser = {h: g.set_index('day').sales_value for h,g in txg.groupby('household_key')}
def lag_fast(h,s,back):
    ss = ser.get(h)
    if ss is None: return 0.0
    idx = ss.index; lo, hi = s+1-back, s+28-back
    sel = idx[(idx>=lo)&(idx<=hi)]
    return float(ss.loc[sel].sum()) if len(sel) else 0.0
lag13 = np.array([lag_fast(h,s,336) for h,s in zip(hk,sd)])
lag1y = np.array([lag_fast(h,s,364) for h,s in zip(hk,sd)])
lag15 = np.array([lag_fast(h,s,392) for h,s in zip(hk,sd)])
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
def prep(cols):
    X=m[cols].astype(float).copy()
    X=X.fillna(X.median())
    mu=X.mean(0); sg=X.std(0)+1e-9
    return (X.values-mu.values)/sg.values
def run(cols, lams=(30,100,300)):
    Xz = prep(cols)
    tr=sd<=347; ev=sd>=375
    out={}
    for lam in lams:
        Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
        w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(cols)), Xt.T@(yt-ym))
        out[lam]=round(float(np.mean(np.abs(Xz[ev]@w+ym-y[ev]))),2)
    return out
print('base E008:', run(feats))
m2 = m.copy()
m2['slag13']=lag13; m2['slag1y']=lag1y; m2['slag15']=lag15
m2['slag13_r'] = lag13/(m2['sp84'].fillna(0)+1)
print('+seasonal:', run(feats+['slag13','slag1y','slag15','slag13_r']))
m2['i_sp28_trend'] = m2['sp28'].fillna(0)*m2['trend_84'].fillna(0)
m2['i_sp28_nact'] = m2['sp28'].fillna(0)*m2['nact28'].fillna(0)
m2['i_sp28_demo4'] = m2['sp28'].fillna(0)*(m2['d_classification_4'].fillna('0')=='5+').astype(float)
m2['i_sp28_kid'] = m2['sp28'].fillna(0)*(m2['d_kid_category_desc'].fillna('None/Unknown').isin(['1','2','3+'])).astype(float)
m2['i_newma_rec'] = m2['newma28'].fillna(0)*m2['days_since_last'].fillna(0)
m2['sq_sp28'] = m2['sp28'].fillna(0)**2
m2['sp28_x_sp84rate'] = m2['sp28'].fillna(0)*m2['sp84_rate'].fillna(0)
print('+interact:', run(feats+['i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))
print('+both:', run(feats+['slag13','slag1y','slag15','slag13_r','i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
hk = m['household_key'].values; sd = m['snapshot_day'].values
v = A.snapshot(459)
txg = v.table('transactions').groupby(['household_key','day'])['sales_value'].sum().reset_index()
ser = {h: g.set_index('day').sales_value for h,g in txg.groupby('household_key')}
def lag_fast(h,s,back):
    ss = ser.get(h)
    if ss is None: return 0.0
    idx = ss.index; lo, hi = s+1-back, s+28-back
    sel = idx[(idx>=lo)&(idx<=hi)]
    return float(ss.loc[sel].sum()) if len(sel) else 0.0
lag13 = np.array([lag_fast(h,s,336) for h,s in zip(hk,sd)])
lag1y = np.array([lag_fast(h,s,364) for h,s in zip(hk,sd)])
lag15 = np.array([lag_fast(h,s,392) for h,s in zip(hk,sd)])
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
def prep(df, cols):
    X=df[cols].astype(float).copy()
    X=X.fillna(X.median())
    mu=X.mean(0); sg=X.std(0)+1e-9
    return (X.values-mu.values)/sg.values
def run(df, cols, lams=(30,100,300)):
    Xz = prep(df, cols)
    tr=sd<=347; ev=sd>=375
    out={}
    for lam in lams:
        Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
        w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(cols)), Xt.T@(yt-ym))
        out[lam]=round(float(np.mean(np.abs(Xz[ev]@w+ym-y[ev]))),2)
    return out
print('base E008:', run(m, feats))
m2 = m.copy()
m2['slag13']=lag13; m2['slag1y']=lag1y; m2['slag15']=lag15
m2['slag13_r'] = lag13/(m2['sp84'].fillna(0)+1)
print('+seasonal:', run(m2, feats+['slag13','slag1y','slag15','slag13_r']))
m2['i_sp28_trend'] = m2['sp28'].fillna(0)*m2['trend_84'].fillna(0)
m2['i_sp28_nact'] = m2['sp28'].fillna(0)*m2['nact28'].fillna(0)
m2['i_sp28_demo4'] = m2['sp28'].fillna(0)*(m2['d_classification_4'].fillna('0')=='5+').astype(float)
m2['i_sp28_kid'] = m2['sp28'].fillna(0)*(m2['d_kid_category_desc'].fillna('None/Unknown').isin(['1','2','3+'])).astype(float)
m2['i_newma_rec'] = m2['newma28'].fillna(0)*m2['days_since_last'].fillna(0)
m2['sq_sp28'] = m2['sp28'].fillna(0)**2
m2['sp28_x_sp84rate'] = m2['sp28'].fillna(0)*m2['sp84_rate'].fillna(0)
print('+interact:', run(m2, feats+['i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))
print('+both:', run(m2, feats+['slag13','slag1y','slag15','slag13_r','i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
sd = m['snapshot_day'].values
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
def prep(df, cols):
    X=df[cols].astype(float).copy(); X=X.fillna(X.median())
    mu=X.mean(0); sg=X.std(0)+1e-9
    return (X.values-mu.values)/sg.values
def run(df, cols, lams=(30,100,300)):
    Xz = prep(df, cols); tr=sd<=347; ev=sd>=375; out={}
    for lam in lams:
        Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
        w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(cols)), Xt.T@(yt-ym))
        out[lam]=round(float(np.mean(np.abs(Xz[ev]@w+ym-y[ev]))),2)
    return out
m2 = m.copy()
m2['i_sp28_trend'] = m2['sp28'].fillna(0)*m2['trend_84'].fillna(0)
m2['i_sp28_nact'] = m2['sp28'].fillna(0)*m2['nact28'].fillna(0)
m2['i_sp28_demo4'] = m2['sp28'].fillna(0)*(m2['d_classification_4'].astype(str)=='5+').astype(float)
m2['i_sp28_kid'] = m2['sp28'].fillna(0)*m2['d_kid_category_desc'].astype(str).isin(['1','2','3+']).astype(float)
m2['i_newma_rec'] = m2['newma28'].fillna(0)*m2['days_since_last'].fillna(0)
m2['sq_sp28'] = m2['sp28'].fillna(0)**2
m2['sp28_x_sp84rate'] = m2['sp28'].fillna(0)*m2['sp84_rate'].fillna(0)
print('+interact:', run(m2, feats+['i_sp28_trend','i_sp28_nact','i_sp28_demo4','i_sp28_kid','i_newma_rec','sq_sp28','sp28_x_sp84rate']))
# also test: dropping weak/duplicate columns (feature pruning) in proxy
bot = ['discshare364','dsh364_GROCE','evshare84','sp_lag1y','splag1y','z_n_windows_hist','tenure','z_tenure_days','topstore','week_mod52','d_classification_2','d_homeowner_desc','topstore']
print('pruned:', run(m2, [c for c in feats if c not in bot]))


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
sd = m['snapshot_day'].values
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
def prep(df, cols):
    X=df[cols].astype(float).copy(); X=X.fillna(X.median())
    mu=X.mean(0); sg=X.std(0)+1e-9
    return (X.values-mu.values)/sg.values
def run(df, cols, lams=(30,100,300)):
    Xz = prep(df, cols); tr=sd<=347; ev=sd>=375; out={}
    for lam in lams:
        Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
        w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(cols)), Xt.T@(yt-ym))
        out[lam]=round(float(np.mean(np.abs(Xz[ev]@w+ym-y[ev]))),2)
    return out
m2 = m.copy(); m2['snap_idx'] = (sd - 95)/28.0
print('base:', run(m2, feats))
print('+snap_idx:', run(m2, feats+['snap_idx']))
# drift check: mean y and mean sp28 by snapshot
g = m.groupby('snapshot_day').agg(ymean=('future_spend_4w','mean'), sp28mean=('sp28','mean'))
g['bias'] = g.ymean - g.sp28mean
print(g.round(1))
# display_mailer structure
v = A.snapshot(459)
dm = v.table('display_mailer')
print(dm.dtypes); print(dm.head(3)); print('weeks', dm.week_no.min(), dm.week_no.max(), 'n', len(dm))
print('display vals', dm.display.value_counts(dropna=False).head())
print('mailer vals', dm.mailer.value_counts(dropna=False).head())


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np, time
t0=time.time()
v = A.snapshot(459)
tx = v.table('transactions')
print('tx rows', len(tx))
dm = v.table('display_mailer')
promo = dm[(dm.display>0)|(dm.mailer!='0')][['product_id','store_id','week_no']]
print('promo rows total', len(promo), 'time', round(time.time()-t0,1))
# commodity-level promo intensity: product->commodity map
p = v.table('products')[['product_id','commodity_desc']]
promo_c = promo.merge(p, on='product_id', how='left')
ci = promo_c.groupby(['commodity_desc','week_no']).size().rename('n_promo').reset_index()
print('commodity-week promo table', ci.shape, 'time', round(time.time()-t0,1))
# household recent spend by commodity for a test snapshot (431)
s=431; wk=(s+8)//7
t = tx[tx.day<=s].copy(); t=t[t.day>s-84]; t['week_no']=(t['day']+8)//7
t=t[t.week_no<=wk]
tc = t.merge(p, on='product_id', how='left').groupby('household_key').apply(
    lambda g: g.merge(ci[ci.week_no.between(wk-12,wk)], on=['commodity_desc','week_no'], how='left')['n_promo'].fillna(0).sum()/max(len(g),1))
print('sample exposure computed', tc.shape, 'time', round(time.time()-t0,1))
print(tc.describe())


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np, time
t0=time.time()
v = A.snapshot(459)
tx = v.table('transactions')
dm = v.table('display_mailer')
dm['disp'] = dm['display'].astype(str).astype(float)
dm['mail'] = dm['mailer'].astype(str)
promo = dm[(dm.disp>0)|(dm.mail!='0')][['product_id','store_id','week_no']]
print('promo rows', len(promo), 'time', round(time.time()-t0,1))
p = v.table('products')[['product_id','commodity_desc']]
promo_c = promo.merge(p, on='product_id', how='left')
ci = promo_c.groupby(['commodity_desc','week_no']).size().rename('n_promo').reset_index()
print('commodity-week promo', ci.shape, 'time', round(time.time()-t0,1))
# household exposure at snapshot 431: share of recent spend in promo-active commodity-weeks
s=431; wk=(s+8)//7
t = tx[(tx.day<=s)&(tx.day>s-84)].copy()
t['week_no']=(t['day']+8)//7
tc = t.merge(p, on='product_id', how='left')
tc = tc.merge(ci[ci.week_no.between(wk-12,wk)], on=['commodity_desc','week_no'], how='left')
tc['promo'] = tc['n_promo'].notna().astype(float)
hh = tc.groupby('household_key').agg(spend=('sales_value','sum'), promo_spend=('sales_value', lambda x: 0))
# simpler: promo_spend = sum of sales where promo==1
g = tc.groupby(['household_key','promo'])['sales_value'].sum().unstack(fill_value=0)
g.columns = ['sp0','sp1'] if 0 in g.columns and 1 in g.columns else list(g.columns)
exp = (g.get(1,0)/ (g.get(0,0)+g.get(1,0)+1e-9)).rename('promo_share')
print('exposure sample', exp.shape, 'time', round(time.time()-t0,1))
print(exp.describe())


# ---- cell ----

import agent_api as A, pandas as pd, numpy as np, time
t0=time.time()
NEW = ['pm_share84','pm_share28','pc_share84','pc_share28','pdisp_share84','pmail_share84','pm_spend84','pm_spend28','p_wks84','snap_idx']

def build(view, s):
    wk = (s + 8) // 7
    tx = view.table('transactions')
    hh = view.households
    if hh is None:
        first = tx.groupby('household_key')['day'].min()
        hh = first[first <= s - 84].index
    if isinstance(hh, pd.DataFrame): hh = hh.index
    hh = pd.Index(hh)
    out = pd.DataFrame(index=hh)
    out['snap_idx'] = (s - 95) / 28.0
    t = tx[tx['day'] > s - 84]
    if len(t) == 0:
        for c in NEW[:-1]: out[c] = 0.0
        return out
    t = t[['household_key','day','product_id','store_id','sales_value']].copy()
    t['week_no'] = ((t['day'] + 8) // 7).astype('int16')
    dm = view.table('display_mailer')
    dm = dm[dm['week_no'] >= wk - 11]
    disp = pd.to_numeric(dm['display'].astype(str), errors='coerce').fillna(0).values
    mail = dm['mailer'].astype(str).values
    is_p = (disp > 0) | (mail != '0')
    pr = pd.DataFrame({'product_id': dm['product_id'].values[is_p],
                       'store_id': dm['store_id'].values[is_p],
                       'week_no': dm['week_no'].values[is_p],
                       'df': (disp[is_p] > 0).astype(np.float32),
                       'mf': np.isin(mail[is_p], ['A','D','H','F']).astype(np.float32)})
    pr = pr.groupby(['product_id','store_id','week_no'], as_index=False).max()
    p = view.table('products')[['product_id','commodity_desc']]
    pc = pr[['product_id','week_no']].drop_duplicates().merge(p, on='product_id', how='left')
    ci = pc.groupby(['commodity_desc','week_no'], as_index=False).size().rename(columns={'size':'npr'})
    tm = t.merge(pr, on=['product_id','store_id','week_no'], how='left')
    tm[['df','mf']] = tm[['df','mf']].fillna(0.0)
    tm['f'] = ((tm['df'] + tm['mf']) > 0).astype(np.float32)
    tc = t.merge(p, on='product_id', how='left').merge(ci, on=['commodity_desc','week_no'], how='left')
    tc['cf'] = tc['npr'].notna().astype(np.float32)
    def S(df, mask): return df.loc[mask].groupby('household_key')['sales_value'].sum()
    tot84 = t.groupby('household_key')['sales_value'].sum()
    tot28 = t[t['day'] > s-28].groupby('household_key')['sales_value'].sum()
    pm84 = S(tm, tm['f']==1); pm28 = S(tm, (tm['f']==1)&(tm['day']>s-28))
    pd84 = S(tm, tm['df']==1); pml84 = S(tm, tm['mf']==1)
    pc84 = S(tc, tc['cf']==1); pc28 = S(tc, (tc['cf']==1)&(tc['day']>s-28))
    wks = tm.loc[tm['f']==1].groupby('household_key')['week_no'].nunique()
    r = lambda x: x.reindex(hh).fillna(0.0)
    out['pm_spend84'] = r(pm84); out['pm_spend28'] = r(pm28)
    t84 = r(tot84).values; t28v = r(tot28).values
    out['pm_share84'] = np.where(t84>0, r(pm84).values/np.maximum(t84,1e-9), 0.0)
    out['pm_share28'] = np.where(t28v>0, r(pm28).values/np.maximum(t28v,1e-9), 0.0)
    out['pdisp_share84'] = np.where(t84>0, r(pd84).values/np.maximum(t84,1e-9), 0.0)
    out['pmail_share84'] = np.where(t84>0, r(pml84).values/np.maximum(t84,1e-9), 0.0)
    out['pc_share84'] = np.where(t84>0, r(pc84).values/np.maximum(t84,1e-9), 0.0)
    out['pc_share28'] = np.where(t28v>0, r(pc28).values/np.maximum(t28v,1e-9), 0.0)
    out['p_wks84'] = (r(wks)/12.0).clip(0,1)
    return out

feat = A.build_features(build)
print('built', feat.shape, 'time', round(time.time()-t0,1))
print(feat[NEW].describe().round(3).to_string())
# proxy ridge: base e008 vs +new
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float); sd = m['snapshot_day'].values
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
m = m.merge(feat, on=['household_key','snapshot_day'], how='left')
def run(cols, lam=100):
    X=m[cols].astype(float); X=X.fillna(X.median()); mu=X.mean(0); sg=X.std(0)+1e-9
    Xz=(X.values-mu.values)/sg.values
    tr=sd<=347; ev=sd>=375; Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
    w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(cols)), Xt.T@(yt-ym))
    return round(float(np.mean(np.abs(Xz[ev]@w+ym-y[ev]))),2)
print('base:', run(feats))
print('+promo:', run(feats+NEW))
path = A.save_table(feat, 'e015_promo')
print('saved', path)
