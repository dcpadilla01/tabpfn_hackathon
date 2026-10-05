import agent_api as A, pandas as pd, numpy as np
print(A.snapshot_days())
tt = A.train_targets()
print('train_targets', tt.shape, tt.columns.tolist())
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']).round(2))
print('zero share', round((tt.future_spend_4w==0).mean(),4))
print(tt.future_spend_4w.describe().round(2))
v = A.snapshot()
tr = v.table('transactions')
print('trans', tr.shape, 'hh', tr.household_key.nunique())
pr = v.table('products'); print('products', pr.shape, pr.department.nunique())
names = ['allF.parquet','e002_features.parquet','e004_features.parquet','e004_new.parquet','e005_newfeats.parquet','f_weekly.parquet','lagfeats.parquet','lagfeats2.parquet','e005_preds.parquet','e004_preds.parquet']
for nm in names:
    try:
        df = A.load_saved(nm)
        print('==',nm, df.shape)
        print(list(df.columns))
    except Exception as e:
        print('==',nm,'ERR',type(e).__name__, e)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
t0=time.time()
print('xgb', xgb.__version__)
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
W2 = W.drop(columns=[ycol], errors='ignore')
F = F.merge(W2, on=['household_key','snapshot_day'], how='left')
print('merged', F.shape)
base_cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
week_cols = [c for c in W2.columns if c not in ('household_key','snapshot_day')]
print('n base', len(base_cols), 'n week', len(week_cols))
print(F[week_cols].isna().mean().round(3).to_string())
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403,431]
Ftr = F[F.snapshot_day.isin(tr_days) & F[ycol].notna()].copy()
print('train rows', Ftr.shape, 'time', round(time.time()-t0,1))
# quick correlation of weekly feats with target
sub = Ftr[week_cols+['snapshot_day']].copy(); sub['y']=Ftr[ycol].values
cor = sub.corr()['y'].drop('y').drop('snapshot_day')
print(cor.round(3).to_string())
print('elapsed', round(time.time()-t0,1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
m = F.merge(W.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
tr = m[m.future_spend_4w.notna()]
for a,b in [('s1','wk_1'),('s2','wk_2'),('s3','wk_3'),('s4','wk_4'),('s2','lag28_56'),('spend_28','wk_1'),('spend_28','s1')]:
    print(a,b,'corr',round(tr[a].corr(tr[b]),4), 'meanabsdiff', round((tr[a]-tr[b]).abs().mean(),3))
print(m[['s1','s2','s3','s4','wk_1','lag28_56','spend_28']].describe().round(2).to_string())
# per-snapshot means of weekly feats to confirm no leakage (computed only from past)
print(m.groupby('snapshot_day')[['wk_1','lag28_56','sp28_yag']].mean().round(2).to_string())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
print(W.columns.tolist())
m = F.merge(W, on=['household_key','snapshot_day'], how='left')
tr = m[m.future_spend_4w.notna()]
for a,b in [('s1','wk_1'),('s2','wk_2'),('s3','wk_3'),('s4','wk_4'),('s2','lag28_56'),('spend_28','wk_1'),('spend_28','s1'),('spend_56','s2')]:
    print(a,b,'corr',round(tr[a].corr(tr[b]),4), 'meanabsdiff', round((tr[a]-tr[b]).abs().mean(),3))
print(m[['s1','s2','s3','s4','wk_1','wk_2','lag28_56','spend_28','spend_56']].describe().round(2).to_string())
print(m.groupby('snapshot_day')[['wk_1','lag28_56','sp28_yag']].mean().round(2).to_string())


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
t0=time.time()
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
M = F.merge(W, on=['household_key','snapshot_day'], how='left')
new_week = ['wk_1','wk_2','wk_3','wk_4','wk_5','wk_6','wk_7','wk_8','wk_13','wk_26','wk_39','wk_52',
            'wk_active_8','dow_0','dow_1','dow_2','dow_3','dow_4','dow_5','dow_6','wkend_share',
            'topstore_share_84','ncommod_84']
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403,431]

def feats(df, extra):
    cols = [c for c in df.columns if c not in ('household_key','snapshot_day',ycol)]
    if extra is not None:
        cols = cols + [c for c in extra if c not in cols]
    return df[cols].astype(float)

def run(Xdf, y, days, w_ref, eval_days, seeds=(7,17,27), lr=0.03, rounds=2400, depth=6, decay=140.0, subs=0.8, cols=0.8):
    maes=[]; preds={}
    for ed in eval_days:
        trm = days < ed
        Xtr, ytr = Xdf[trm], y[trm]
        w = 0.5**((w_ref - days[trm])/decay)
        ps = np.zeros((~trm).sum(), dtype=float)
        for s in seeds:
            m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=lr, max_depth=depth,
                objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
                subsample=subs, colsample_bytree=cols, random_state=s, n_jobs=32)
            m.fit(Xtr, ytr, sample_weight=w)
            ps += m.predict(Xdf[~trm])/len(seeds)
        yy = y[~trm]
        maes.append(np.abs(ps-yy).mean()); preds[ed]=ps
    return maes, preds

base_cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
for name, extra in [('BASE', None), ('BASE+WEEK', new_week)]:
    X = feats(M, extra).values.astype(np.float32)
    y = M[ycol].values.astype(float); d = M.snapshot_day.values
    maes,_ = run(X, y, d, 431, [403,431])
    print(name, 'MAE@403,431:', [round(m,3) for m in maes], 'time', round(time.time()-t0,1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
M = F.merge(W, on=['household_key','snapshot_day'], how='left')
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403,431]
base_cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = M[base_cols].values.astype(np.float32)
y = M[ycol].values.astype(float); d = M.snapshot_day.values
t0=time.time()
m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, objective='reg:quantileerror',
    quantile_alpha=0.5, tree_method='hist', subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=-1)
mask = d < 431
w = 0.5**((431 - d[mask])/140.0)
m.fit(X[mask], y[mask], sample_weight=w)
print('fit 1200 rounds, 24k rows:', round(time.time()-t0,1),'s')
p = m.predict(X[~mask]); print('MAE431', round(np.abs(p-y[~mask]).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
M = F.merge(W, on=['household_key','snapshot_day'], how='left')
base_cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
new_week = ['wk_1','wk_2','wk_3','wk_4','wk_5','wk_6','wk_7','wk_8','wk_13','wk_26','wk_39','wk_52',
            'wk_active_8','dow_0','dow_1','dow_2','dow_3','dow_4','dow_5','dow_6','wkend_share',
            'topstore_share_84','ncommod_84']
y = M[ycol].values.astype(float); d = M.snapshot_day.values

def cv(cols_list, eval_days=(403,431), seeds=(7,17), rounds=1200, lr=0.03, depth=6, decay=140.0):
    X = M[cols_list].values.astype(np.float32)
    out={}
    for ed in eval_days:
        trm = (d < ed) & ~np.isnan(y)
        tem = (d == ed) & ~np.isnan(y)
        w = 0.5**((ed - d[trm])/decay)
        ps = np.zeros(tem.sum())
        for s in seeds:
            m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=lr, max_depth=depth,
                objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
                subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1)
            m.fit(X[trm], y[trm], sample_weight=w)
            ps += m.predict(X[tem])/len(seeds)
        out[ed]=round(np.abs(ps - y[tem]).mean(),3)
    return out

t0=time.time()
r1 = cv(base_cols); print('BASE', r1, round(time.time()-t0,1))
r2 = cv(base_cols+new_week); print('BASE+WEEK', r2, round(time.time()-t0,1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet')
ycol='future_spend_4w'
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = F[cols].values.astype(np.float32)
y = F[ycol].values.astype(float); d = F.snapshot_day.values

def fit_pred(Xtr, ytr, w, Xte, seed, obj='q', rounds=1200, lr=0.03, depth=6):
    kw = dict(n_estimators=rounds, learning_rate=lr, max_depth=depth, tree_method='hist',
              subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    if obj=='q':
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, **kw)
    else:
        m = xgb.XGBRegressor(objective='reg:absoluteerror', **kw)
    m.fit(Xtr, ytr, sample_weight=w)
    return m.predict(Xte)

def cv(transform, obj='q', eval_days=(403,431), seeds=(7,17), inv=None):
    out={}
    for ed in eval_days:
        trm = (d < ed) & ~np.isnan(y); tem = (d == ed) & ~np.isnan(y)
        w = 0.5**((ed - d[trm])/140.0)
        yt = transform(y[trm])
        ps = np.zeros(tem.sum())
        for s in seeds:
            p = fit_pred(X[trm], yt, w, X[tem], s, obj=obj)
            ps += (inv(p) if inv else p)/len(seeds)
        out[ed]=round(np.abs(ps - y[tem]).mean(),3)
    return out

t0=time.time()
idn = lambda v: v
print('raw-q   ', cv(idn), round(time.time()-t0,1))
print('log-q   ', cv(np.log1p, inv=np.expm1), round(time.time()-t0,1))
print('raw-ae  ', cv(idn, obj='ae'), round(time.time()-t0,1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet'); e5 = A.load_saved('e005_preds.parquet')
ycol='future_spend_4w'
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = F[cols].values.astype(np.float32)
y = F[ycol].values.astype(float); d = F.snapshot_day.values
valm = ~np.isnan(y)
t0=time.time()
def qfit(Xtr,ytr,w,Xte,seed,rounds,depth=6):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=depth,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
trm = (d<459)&~np.isnan(y); w = 0.5**((459-d[trm])/140.0)
p13 = np.zeros(valm.sum())
for s in (7,17,27):
    p13 += qfit(X[trm],y[trm],w,X[valm],s,2400)/3
print('repro-all13 done', round(time.time()-t0,1))
trm11 = (d<403)&~np.isnan(y); w11 = 0.5**((459-d[trm11])/140.0)
p11 = np.zeros(valm.sum())
for s in (7,17,27):
    p11 += qfit(X[trm11],y[trm11],w11,X[valm],s,2400)/3
print('repro-11 done', round(time.time()-t0,1))
e5v = e5.prediction.values
for nm,p in [('all13',p13),('excl403/431',p11)]:
    print(nm,'mean',round(p.mean(),2),'| e5 mean',round(e5v.mean(),2),'| corr',round(np.corrcoef(p,e5v)[0,1],4),'| MAEdiff',round(np.abs(p-e5v).mean(),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet'); e5 = A.load_saved('e005_preds.parquet')
ycol='future_spend_4w'
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = F[cols].values.astype(np.float32)
y = F[ycol].values.astype(float); d = F.snapshot_day.values
valm = (d>=459)
t0=time.time()
def qfit(Xtr,ytr,w,Xte,seed,rounds,depth=6):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=depth,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
res={}
for nm, trmask in [('all13',(d<459)&~np.isnan(y)), ('excl403_431',(d<403)&~np.isnan(y))]:
    trm = trmask; w = 0.5**((459 - d[trm])/140.0)
    p = np.zeros(valm.sum())
    for s in (7,17,27):
        p += qfit(X[trm],y[trm],w,X[valm],s,2400)/3
    res[nm]=p
    print(nm,'done',round(time.time()-t0,1),'mean',round(p.mean(),2))
e5v = e5.prediction.values
for nm,p in res.items():
    print(nm,'| corr to e5',round(np.corrcoef(p,e5v)[0,1],4),'| MAEdiff',round(np.abs(p-e5v).mean(),3))
out = F.loc[valm,['household_key','snapshot_day']].copy()
out['p13']=res['all13']; out['p11']=res['excl403_431']
A.save_table(out,'repro_e5.parquet')
print('saved', out.shape)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet')
ycol='future_spend_4w'
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
X = F[cols].values.astype(np.float32)
y = F[ycol].values.astype(float); d = F.snapshot_day.values
t0=time.time()
def qfit(Xtr,ytr,w,Xte,seed,rounds=1200,depth=6):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=depth,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
def ev(ed, cut, seeds=(7,17)):
    trm = (d<cut)&~np.isnan(y); tem = (d==ed)
    w = 0.5**((ed - d[trm])/140.0)
    p = np.zeros(tem.sum())
    for s in seeds: p += qfit(X[trm],y[trm],w,X[tem],s)/len(seeds)
    ya = y[tem]
    return np.abs(p-ya).mean(), p.mean(), ya.mean(), p
for ed in (403,431):
    for nm,cut in [('all',(ed-28+1)),('excl2',(ed-84+1))]:
        mae,pm,yam,_ = ev(ed,cut)
        print(f'day {ed} cut<={cut-1} [{nm}] MAE {mae:.3f} predmean {pm:.1f} actualmean {yam:.1f} bias {pm-yam:+.1f}')
print('elapsed',round(time.time()-t0,1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
F = A.load_saved('allF.parquet')
ycol='future_spend_4w'
allc = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
wf = ['week_of_year','week_sin','week_cos']
variants = {'FULL': allc, 'NOWEEK': [c for c in allc if c not in wf],
            'NOSINCOS': [c for c in allc if c not in ('week_sin','week_cos')],
            'ONLYSINCOS': [c for c in allc if c not in ('week_of_year',)]}
y = F[ycol].values.astype(float); d = F.snapshot_day.values
def qfit(Xtr,ytr,w,Xte,seed,rounds=1200):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=6,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
t0=time.time()
# extrapolation proxy: train <= 347, eval 375/403/431 (unseen raw weeks)
for nm, cols in variants.items():
    X = F[cols].values.astype(np.float32)
    maes=[]
    for ed in (375,403,431):
        trm = (d<=347)&~np.isnan(y); tem=(d==ed)
        w = 0.5**((ed-d[trm])/140.0)
        p = qfit(X[trm],y[trm],w,X[tem],7)
        maes.append(round(np.abs(p-y[tem]).mean(),2))
    print('EXTRAP', nm, maes, round(time.time()-t0,1))
# in-range sanity: train < 403, eval 403; train < 431, eval 431
for nm, cols in variants.items():
    X = F[cols].values.astype(np.float32)
    maes=[]
    for ed in (403,431):
        trm = (d<ed)&~np.isnan(y); tem=(d==ed)
        w = 0.5**((ed-d[trm])/140.0)
        p = qfit(X[trm],y[trm],w,X[tem],7)
        maes.append(round(np.abs(p-y[tem]).mean(),2))
    print('INRANGE', nm, maes, round(time.time()-t0,1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
t0=time.time()
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
M = F.merge(W[['household_key','snapshot_day','sp28_yag','sp84_yag']], on=['household_key','snapshot_day'], how='left')
cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
# year-ago pace ratios (NaN when no year-ago history; XGBoost handles NaN)
M['r28yag'] = np.where(M.sp28_yag>0, M.spend_28/M.sp28_yag, np.nan)
M['r84yag'] = np.where(M.sp84_yag>0, M.spend_84/M.sp84_yag, np.nan)
M['r28yag_c'] = M.r28yag.clip(0,5); M['r84yag_c'] = M.r84yag.clip(0,5)
colsR = cols + ['r28yag','r84yag','r28yag_c','r84yag_c']
y = M[ycol].values.astype(float); d = M.snapshot_day.values
tr_days_all = [95,123,151,179,207,235,263,291,319,347,375,403,431]
def qfit(Xtr,ytr,w,Xte,seed,rounds):
    m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=0.03, max_depth=6,
        objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist',
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    m.fit(Xtr,ytr,sample_weight=w); return m.predict(Xte)
# local CV at 403/431 (1 seed, 1200 rounds)
for nm, cl in [('BASE',cols), ('RATIO',colsR)]:
    X = M[cl].values.astype(np.float32); maes=[]
    for ed in (403,431):
        trm=(d<ed)&~np.isnan(y); tem=(d==ed); w=0.5**((ed-d[trm])/140.0)
        p=qfit(X[trm],y[trm],w,X[tem],7,1200)
        maes.append(round(np.abs(p-y[tem]).mean(),3))
    print(nm, maes, round(time.time()-t0,1))
# final: E005 recipe (all13, decay140, 2400 rounds, 3 seeds) on RATIO features -> val preds
X = M[colsR].values.astype(np.float32)
valm = (d>=459); trm=(d<459)&~np.isnan(y)
w = 0.5**((459-d[trm])/140.0)
pv = np.zeros(valm.sum())
for s in (7,17,27): pv += qfit(X[trm],y[trm],w,X[valm],s,2400)/3
out = M.loc[valm,['household_key','snapshot_day']].copy(); out['prediction']=pv
A.save_table(out,'e010_preds.parquet')
print('saved e010', out.shape, 'predmean', round(pv.mean(),2), 'elapsed', round(time.time()-t0,1))
