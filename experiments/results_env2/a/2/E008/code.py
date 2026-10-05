
import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb

feats = api.load_saved('feats_v3.parquet')
print('feats', feats.shape)
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
print('rows', len(df), 'n_feat', len(feat_cols), 'days', sorted(df.snapshot_day.unique()))
obj_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(df[c])]
print('non-numeric cols:', obj_cols[:10])
for c in obj_cols:
    if str(df[c].dtype) == 'category': df[c] = df[c].cat.codes
    else: df[c] = pd.factorize(df[c])[0]

holdout = [347, 375, 403, 431]
inner = [d for d in sorted(df.snapshot_day.unique()) if d not in holdout]
m_tr = df.snapshot_day.isin(inner).values
m_ho = df.snapshot_day.isin(holdout).values
Xtr = df.loc[m_tr, feat_cols].astype(float).values
Xho = df.loc[m_ho, feat_cols].astype(float).values
ytr = df.loc[m_tr,'future_spend_4w'].values
yho = df.loc[m_ho,'future_spend_4w'].values
print('train', Xtr.shape, 'holdout', Xho.shape, 'holdout zero frac', round((yho==0).mean(),3))

base = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8)

def fit_pred(params, Xtr, ytr, Xho, post=None):
    m = xgb.XGBRegressor(**params); m.fit(Xtr, ytr)
    p = m.predict(Xho)
    if post is not None: p = post(p)
    return np.clip(p, 0, None)

t0=time.time()
oof_sq = fit_pred({**base,'objective':'reg:squarederror'}, Xtr, ytr, Xho)
print('sq  MAE', round(np.abs(oof_sq-yho).mean(),3), round(time.time()-t0,1),'s')

t0=time.time()
oof_med = fit_pred({**base,'n_estimators':1500,'learning_rate':0.03,'max_depth':6,
                    'objective':'reg:quantileerror','quantile_alpha':0.5}, Xtr, ytr, Xho)
print('med MAE', round(np.abs(oof_med-yho).mean(),3), round(time.time()-t0,1),'s')

t0=time.time()
oof_log = fit_pred({**base,'objective':'reg:squarederror'}, Xtr, np.log1p(ytr), Xho, post=np.expm1)
print('log MAE', round(np.abs(oof_log-yho).mean(),3), round(time.time()-t0,1),'s')

print('e006-approx MAE', round(np.abs(0.5*oof_sq+0.5*oof_med-yho).mean(),3))
rows=[]
for a in np.arange(0,1.0001,0.05):
    for b in np.arange(0,1.0001-a+1e-9,0.05):
        c = 1-a-b
        if c < -1e-9: continue
        mae = np.abs(a*oof_sq+b*oof_med+c*oof_log-yho).mean()
        rows.append((round(a,2),round(b,2),round(c,2),round(mae,3)))
rows.sort(key=lambda r:r[3])
print('top weight combos (sq,med,log,mae):')
for r in rows[:12]: print(r)

oof_df = pd.DataFrame({'household_key': df.loc[m_ho,'household_key'].values,
                       'snapshot_day': df.loc[m_ho,'snapshot_day'].values,
                       'oof_sq':oof_sq,'oof_med':oof_med,'oof_log':oof_log})
print('saved:', api.save_table(oof_df,'oof_e008'))

for n in ['pred_e004','pred_e005','pred_e006']:
    p = api.load_saved(n)
    print(n, p.shape, p.columns.tolist())


# ---- cell ----

import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
holdout = [347, 375, 403, 431]
inner = [d for d in sorted(df.snapshot_day.unique()) if d not in holdout]
m_tr = df.snapshot_day.isin(inner).values
m_ho = df.snapshot_day.isin(holdout).values
Xtr = df.loc[m_tr, feat_cols].astype(float).values
Xho = df.loc[m_ho, feat_cols].astype(float).values
ytr = df.loc[m_tr,'future_spend_4w'].values
yho = df.loc[m_ho,'future_spend_4w'].values

base = dict(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8)
def mae(p): return round(np.abs(p-yho).mean(),3)

# reference: median model
t0=time.time()
m = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr,ytr)
p_med = np.clip(m.predict(Xho),0,None)
print('med ref MAE', mae(p_med), round(time.time()-t0,1),'s')

# A) winsorized target (median objective)
for q in [0.95, 0.98, 0.99]:
    cap = np.quantile(ytr, q)
    m = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr, np.minimum(ytr,cap))
    print(f'winsor@{q} (cap={cap:.0f}) MAE', mae(np.clip(m.predict(Xho),0,None)))

# B) two-part model
t0=time.time()
clf = xgb.XGBClassifier(**{**base,'objective':'binary:logistic','eval_metric':'logloss'}).fit(Xtr,(ytr>0).astype(int))
p_pos = clf.predict_proba(Xho)[:,1]
reg = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr[ytr>0], ytr[ytr>0])
p_amt = np.clip(reg.predict(Xho),0,None)
p2 = p_pos*p_amt
print('two-part MAE', mae(p2), 'pos-rate pred', round(p_pos.mean(),3), 'actual', round((yho>0).mean(),3), round(time.time()-t0,1),'s')
# two-part with calibrated exponent on p_pos
for e in [0.7,0.85,1.15]:
    print(f'  two-part p^{e} MAE', mae((p_pos**e)*p_amt))
# blend two-part with median model
for w in [0.2,0.3,0.4,0.5]:
    print(f'  blend med+{w}*twopart MAE', mae((1-w)*p_med + w*p2))

# C) global shrink of median model
for s in [0.9,0.95,1.05,1.1]:
    print(f'shrink {s} MAE', mae(p_med*s))


# ---- cell ----

import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
holdout = [347, 375, 403, 431]
inner = [d for d in sorted(df.snapshot_day.unique()) if d not in holdout]
m_tr = df.snapshot_day.isin(inner).values
m_ho = df.snapshot_day.isin(holdout).values
Xtr = df.loc[m_tr, feat_cols].astype(float).values
Xho = df.loc[m_ho, feat_cols].astype(float).values
ytr = df.loc[m_tr,'future_spend_4w'].values
yho = df.loc[m_ho,'future_spend_4w'].values
def mae(p): return round(np.abs(np.clip(p,0,None)-yho).mean(),3)

base = dict(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8)
t0=time.time()
mx = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr,ytr)
p_xgb = mx.predict(Xho); print('xgb med', mae(p_xgb), round(time.time()-t0,1),'s')

t0=time.time()
h1 = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=600, learning_rate=0.05,
     max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=7).fit(Xtr,ytr)
p_h1 = h1.predict(Xho); print('histgb quantile', mae(p_h1), round(time.time()-t0,1),'s')
t0=time.time()
h2 = HistGradientBoostingRegressor(loss='absolute_error', max_iter=600, learning_rate=0.05,
     max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=7).fit(Xtr,ytr)
p_h2 = h2.predict(Xho); print('histgb abserr', mae(p_h2), round(time.time()-t0,1),'s')

for w in [0.2,0.3,0.4,0.5]:
    print(f'blend xgb+{w}*hist_q MAE', mae((1-w)*p_xgb + w*p_h1))
for w in [0.2,0.3,0.4]:
    print(f'blend xgb+{w}*(h1+h2)/2 MAE', mae((1-w)*p_xgb + w*0.5*(p_h1+p_h2)))

# feature selection by gain importance
imp = mx.feature_importances_
order = np.argsort(-imp)
for K in [30, 50, 70]:
    cols = [feat_cols[i] for i in order[:K]]
    m = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr[:,order[:K]],ytr)
    print(f'top{K} feat MAE', mae(m.predict(Xho[:,order[:K]])))
print('top10:', [feat_cols[i] for i in order[:10]])


# ---- cell ----

import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
holdout = [347, 375, 403, 431]
inner = [d for d in sorted(df.snapshot_day.unique()) if d not in holdout]
m_tr = df.snapshot_day.isin(inner).values
m_ho = df.snapshot_day.isin(holdout).values
Xtr = df.loc[m_tr, feat_cols].astype(float).values
Xho = df.loc[m_ho, feat_cols].astype(float).values
ytr = df.loc[m_tr,'future_spend_4w'].values
yho = df.loc[m_ho,'future_spend_4w'].values
def mae(p): return round(np.abs(np.clip(p,0,None)-yho).mean(),3)

# tune HistGB quantile
best=(None,1e9)
t0=time.time()
for mi in [400, 800, 1200]:
    for lr in [0.03, 0.06]:
        for msl in [25, 60]:
            h = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=mi, learning_rate=lr,
                 max_leaf_nodes=31, min_samples_leaf=msl, l2_regularization=1.0, random_state=7).fit(Xtr,ytr)
            e = mae(h.predict(Xho))
            if e < best[1]: best=((mi,lr,msl),e)
            print(mi,lr,msl,'->',e, round(time.time()-t0,1),'s')
print('BEST histgb', best)

# xgb components (fixed configs from E004/E005)
t0=time.time()
mx = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=10,
     subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8,
     objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr,ytr)
p_xgb = mx.predict(Xho); print('xgb med', mae(p_xgb), round(time.time()-t0,1),'s')
t0=time.time()
ms = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
     subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8,
     objective='reg:squarederror').fit(Xtr,ytr)
p_sq = ms.predict(Xho); print('xgb sq', mae(p_sq), round(time.time()-t0,1),'s')

mi,lr,msl = best[0]
h1 = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=mi, learning_rate=lr,
     max_leaf_nodes=31, min_samples_leaf=msl, l2_regularization=1.0, random_state=7).fit(Xtr,ytr)
p_h = h1.predict(Xho); print('histgb best', mae(p_h))

# weight search over 3 components
rows=[]
for a in np.arange(0,1.001,0.1):
    for b in np.arange(0,1.001-a,0.1):
        c = 1-a-b
        if c < -1e-9: continue
        rows.append((round(a,2),round(b,2),round(c,2), round(mae(a*p_xgb+b*p_sq+c*p_h),3)))
rows.sort(key=lambda r:r[3])
print('top (xgbmed,xgbsq,histgb):')
for r in rows[:10]: print(r)


# ---- cell ----

import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
tr = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
va = feats[~feats.set_index(['household_key','snapshot_day']).index.isin(
        tr.set_index(['household_key','snapshot_day']).index)].copy()
print('train rows', len(tr), 'val rows', len(va))

holdout = [347, 375, 403, 431]
inner = [d for d in sorted(tr.snapshot_day.unique()) if d not in holdout]
mi_tr = tr.snapshot_day.isin(inner).values
mi_ho = tr.snapshot_day.isin(holdout).values
Xa = tr.loc[mi_tr, feat_cols].astype(float).values
Xb = tr.loc[mi_ho, feat_cols].astype(float).values
ya = tr.loc[mi_tr,'future_spend_4w'].values
yb = tr.loc[mi_ho,'future_spend_4w'].values
def mae(p, y): return np.abs(np.clip(p,0,None)-y).mean()

cfg_med = dict(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=10,
               subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8,
               objective='reg:quantileerror', quantile_alpha=0.5)
cfg_sq  = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
               subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8,
               objective='reg:squarederror')
def histgb(seed):
    return HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=400,
        learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=25,
        l2_regularization=1.0, random_state=seed)

t0=time.time()
# inner-holdout OOF for weight selection
o1 = xgb.XGBRegressor(**cfg_med).fit(Xa,ya).predict(Xb)
o2 = xgb.XGBRegressor(**cfg_sq).fit(Xa,ya).predict(Xb)
o3 = 0.5*(histgb(7).fit(Xa,ya).predict(Xb) + histgb(13).fit(Xa,ya).predict(Xb))
print('inner MAEs:', round(mae(o1,yb),3), round(mae(o2,yb),3), round(mae(o3,yb),3), round(time.time()-t0,1),'s')

best=(None,1e9)
for a in np.arange(0.15,0.71,0.05):
    for b in np.arange(0.15,0.71-a,0.05):
        c = 1-a-b
        if c < 0.149: continue
        e = mae(a*o1+b*o2+c*o3, yb)
        if e < best[1]: best=((round(a,2),round(b,2),round(c,2)), e)
w = best[0]; print('chosen weights (med,sq,hist):', w, 'inner MAE', round(best[1],3))

# refit on ALL train snapshots, predict validation
Xtr = tr[feat_cols].astype(float).values; ytr = tr['future_spend_4w'].values
Xva = va[feat_cols].astype(float).values
t0=time.time()
p1 = xgb.XGBRegressor(**cfg_med).fit(Xtr,ytr).predict(Xva)
p2 = xgb.XGBRegressor(**cfg_sq).fit(Xtr,ytr).predict(Xva)
p3 = 0.5*(histgb(7).fit(Xtr,ytr).predict(Xva) + histgb(13).fit(Xtr,ytr).predict(Xva))
pred = np.clip(w[0]*p1 + w[1]*p2 + w[2]*p3, 0, None)
out = va[['household_key','snapshot_day']].copy()
out['prediction'] = pred
print('pred stats: mean', round(pred.mean(),2), 'zero frac', round((pred==0).mean(),3), round(time.time()-t0,1),'s')
path = api.save_table(out, 'pred_e008')
print('saved', path, out.shape)
