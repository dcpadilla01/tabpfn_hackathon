
import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
print("feats shape:", feats.shape)
print("feats cols:", feats.columns.tolist())
m = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
print("merged:", m.shape)
t = m['future_spend_4w']
print(t.describe())
print("zero frac:", (t==0).mean(), "| <10 frac:", (t<10).mean())
print(m.groupby('snapshot_day')['future_spend_4w'].agg([('mean','mean'),('med','median'),('zero', lambda s:(s==0).mean())]))
num = feats.drop(columns=['household_key','snapshot_day']).select_dtypes(include=[np.number])
corrs = num.corrwith(m['future_spend_4w']).sort_values()
print("corr with target (bottom 15):"); print(corrs.head(15))
print("corr with target (top 15):"); print(corrs.tail(15))
print("NaN counts>0:", num.isna().sum()[num.isna().sum()>0].to_dict())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
print("feats dup rows:", feats.duplicated(['household_key','snapshot_day']).sum())
print("feats nunique hh:", feats.household_key.nunique(), "snap days:", sorted(feats.snapshot_day.unique()))
print("tt rows:", len(tt), "tt dup:", tt.duplicated(['household_key','snapshot_day']).sum())
# manual check on one snapshot
d = 95
f95 = feats[feats.snapshot_day==d].set_index('household_key')
t95 = tt[tt.snapshot_day==d].set_index('household_key')['future_spend_4w']
common = f95.index.intersection(t95.index)
print("snap95: n feat", len(f95), "n target", len(t95), "common", len(common))
print("corr spend_28:", np.corrcoef(f95.loc[common,'spend_28'], t95.loc[common])[0,1])
print("corr spend_84:", np.corrcoef(f95.loc[common,'spend_84'], t95.loc[common])[0,1])
print("corr exp4w_blend:", np.corrcoef(f95.loc[common,'exp4w_blend'], t95.loc[common])[0,1])
# check whether same household appears at multiple snapshot days with same feature values
sub = feats[feats.household_key==feats.household_key.iloc[0]][['snapshot_day','spend_28','spend_84','exp4w_blend']]
print(sub)


# ---- cell ----

import agent_api, pandas as pd, numpy as np
pred = agent_api.load_saved('pred_e011.parquet')
tt = agent_api.train_targets()
feats = agent_api.load_saved('feats_v4.parquet')
val = pred[pred.snapshot_day>=459]
mv = val.merge(tt, on=['household_key','snapshot_day'])
print("val rows:", len(mv), "MAE:", np.abs(mv.prediction-mv.future_spend_4w).mean())
mv['err'] = mv.prediction - mv.future_spend_4w
mv['abserr'] = mv.err.abs()
# error by target bucket
mv['tb'] = pd.cut(mv.future_spend_4w, [-1,0.01,25,75,150,300,600,1e9])
print(mv.groupby('tb', observed=True).agg(n=('abserr','size'), mae=('abserr','mean'), bias=('err','mean'), med_pred=('prediction','median')))
# how much MAE from top decile of predictions
q = mv.prediction.quantile(0.9)
hi = mv[mv.prediction>q]
print("top-decile-pred share of total abs err:", hi.abserr.sum()/mv.abserr.sum(), "| their MAE:", hi.abserr.mean(), "| bias:", hi.err.mean())
# zero-target rows
z = mv[mv.future_spend_4w==0]
print("zero-target: n", len(z), "MAE", z.abserr.mean(), "mean pred", z.prediction.mean(), "median pred", z.prediction.median())
# clip test
for cap in [400, 600, 800, 1000, 1200]:
    p = mv.prediction.clip(upper=cap)
    print(f"clip@{cap}: MAE {np.abs(p-mv.future_spend_4w).mean():.3f}")
# global scaling
for s in [0.9, 0.95, 1.0, 1.05]:
    print(f"scale {s}: MAE {np.abs(mv.prediction*s-mv.future_spend_4w).mean():.3f}")
# blend with exp4w_blend at various weights
mv2 = mv.merge(feats[['household_key','snapshot_day','exp4w_blend']], on=['household_key','snapshot_day'])
for w in [0.0,0.2,0.3,0.4,0.5]:
    print(f"w_model {w}: MAE {np.abs(w*mv2.prediction+(1-w)*mv2.exp4w_blend-mv2.future_spend_4w).mean():.3f}")


# ---- cell ----

import agent_api, pandas as pd, numpy as np
pred = agent_api.load_saved('pred_e011.parquet')
tt = agent_api.train_targets()
print(pred.dtypes)
print(pred.head(3))
print(tt.dtypes)
print(tt.head(3))
print("pred snap days:", sorted(pred.snapshot_day.unique()))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS], tr.future_spend_4w
Xev, yev = ev[FCOLS], ev.future_spend_4w
print("train rows", len(tr), "eval rows", len(ev), "eval MAE of median const:", np.abs(np.median(ytr)-yev).mean())

def fit(params, Xtr, ytr, num=400):
    mdl = xgb.XGBRegressor(n_estimators=num, tree_method='hist', n_jobs=4, verbosity=0, **params)
    mdl.fit(Xtr, ytr)
    return mdl

base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
mdl = fit(base, Xtr, ytr)
p = mdl.predict(Xev)
print("A ref quantile: MAE", np.abs(p-yev).mean())
for cap in [400,500,600,800]:
    print(f"  clip@{cap}:", round(np.abs(np.clip(p,None,cap)-yev).mean(),3))

# C: log1p target quantile
mdlC = fit(base, Xtr, np.log1p(ytr))
pC = np.expm1(mdlC.predict(Xev))
print("C log1p quantile: MAE", np.abs(pC-yev).mean())
for cap in [400,600]:
    print(f"  clip@{cap}:", round(np.abs(np.clip(pC,None,cap)-yev).mean(),3))

# D: two-part: classifier gate * median pred
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, min_child_weight=40, subsample=0.9, colsample_bytree=0.8, tree_method='hist', n_jobs=4, verbosity=0)
clf.fit(Xtr, (ytr>0).astype(int))
gate = clf.predict_proba(Xev)[:,1]
print("D two-part gate*pred: MAE", np.abs(gate*p-yev).mean(), "| gate*clip600:", np.abs(gate*np.clip(p,None,600)-yev).mean())

# E: blend with exp4w_blend
eb = ev.set_index(['household_key'])['exp4w_blend']
pe = pd.Series(p, index=ev.household_key)
for w in [0.7,0.8,0.9]:
    print(f"E blend w={w}:", round(np.abs(w*p+(1-w)*eb.values-yev).mean(),3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS], tr.future_spend_4w
Xev, yev = ev[FCOLS], ev.future_spend_4w
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
p = pd.Series(mdl.predict(Xev), index=ev.household_key)
e = pd.DataFrame({'y':yev.values,'p':p.values,'hh':ev.household_key.values,'eb':ev.exp4w_blend.values})
e['err']=e.p-e.y; e['abserr']=e.err.abs()
e['tb']=pd.cut(e.y,[-1,0.01,25,75,150,300,600,1e9])
print(e.groupby('tb',observed=True).agg(n=('abserr','size'),mae=('abserr','mean'),bias=('err','mean'),medp=('p','median')))
# per-household: how much of MAE is within-household volatility vs between-household?
hh_mean_y = tr.groupby('household_key').future_spend_4w.mean()
e['hhmu'] = e.hh.map(hh_mean_y)
print("MAE predict hh train-mean:", np.abs(e.hhmu-e.y).mean())
print("MAE predict exp4w_blend:", np.abs(e.eb-e.y).mean())
print("MAE predict 0.5*model+0.5*hhmu:", np.abs((0.5*e.p+0.5*e.hhmu)-e.y).mean())
print("MAE predict 0.7*model+0.3*hhmu:", np.abs((0.7*e.p+0.3*e.hhmu)-e.y).mean())
print("MAE predict 0.85*model+0.15*hhmu:", np.abs((0.85*e.p+0.15*e.hhmu)-e.y).mean())
# shrink toward household mean of PAST 4w spends (multiple lags): use feats cols lag1_spend..? use exp4w_all as hh long-run
print("MAE predict exp4w_all:", np.abs(ev.set_index('household_key')['exp4w_all'].values-e.y).mean())
# blend model with exp4w_all
ea = ev.set_index('household_key')['exp4w_all'].values
for w in [0.7,0.8,0.9]:
    print(f"blend model+exp4w_all w={w}:", round(np.abs(w*e.p+(1-w)*ea-e.y).mean(),3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS], tr.future_spend_4w
Xev, yev = ev[FCOLS], ev.future_spend_4w
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
ptr = mdl.predict(Xtr); pev = mdl.predict(Xev)

# calibration curve: median y per prediction bin
for X_, p_, y_, nm in [(Xtr,ptr,ytr,'train'),(Xev,pev,yev,'eval')]:
    b = pd.DataFrame({'p':p_,'y':y_})
    b['bin'] = pd.qcut(b.p, 10, duplicates='drop')
    print(nm); print(b.groupby('bin',observed=True).agg(med_p=('p','median'), med_y=('y','median'), mean_y=('y','mean'), n=('y','size')))

# per-household multiplicative calibration from train
tr2 = pd.DataFrame({'hh':tr.household_key.values,'p':ptr,'y':ytr.values})
cal = tr2.groupby('hh').agg(mu_p=('p','mean'), mu_y=('y','mean'))
cal['ratio'] = (cal.mu_y/np.maximum(cal.mu_p,1)).clip(0.5, 3.0)
e = pd.DataFrame({'hh':ev.household_key.values,'p':pev,'y':yev.values})
e = e.merge(cal, left_on='hh', right_index=True, how='left')
e['pcal'] = e.p*e.ratio.fillna(1)
print("MAE pcal (hh ratio calib):", np.abs(e.pcal-e.y).mean())
print("MAE p:", np.abs(e.p-e.y).mean())
# additive version
cal['diff'] = cal.mu_y-cal.mu_p
e2 = e.merge(cal['diff'], left_on='hh', right_index=True, how='left')
e2['pcal2'] = (e2.p+e2['diff']).clip(lower=0)
print("MAE pcal2 (hh additive):", np.abs(e2.pcal2-e2.y).mean())
# isotonic calibration on train preds
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(ptr, ytr)
print("MAE isotonic:", np.abs(iso.predict(pev)-yev).mean())


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS], tr.future_spend_4w
Xev, yev = ev[FCOLS], ev.future_spend_4w
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)

# in-sample MAE to gauge underfit
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
print("in-sample MAE:", np.abs(mdl.predict(Xtr)-ytr).mean(), "| eval MAE:", np.abs(mdl.predict(Xev)-yev).mean())

# 1. seed-bagged median ensemble
preds=[]
for s in [1,2,3]:
    p_ = dict(base); p_['subsample']=0.7+0.1*s; 
    mm = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, random_state=s, **p_).fit(Xtr,ytr)
    preds.append(pd.Series(mm.predict(Xev), index=ev.household_key))
P = pd.concat(preds, axis=1)
print("bag mean MAE:", np.abs(P.mean(1).values-yev).mean(), "| bag median MAE:", np.abs(P.median(1).values-yev).mean())

# 2. residual on blend
eb_tr = tr.exp4w_blend.values; eb_ev = ev.exp4w_blend.values
mr = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr-eb_tr)
pr = mr.predict(Xev)+eb_ev
print("residual-on-blend MAE:", np.abs(pr-yev).mean())

# 3. lower lr, more trees
for ne,lr,dp in [(800,0.04,5),(600,0.05,4),(500,0.06,6)]:
    p_=dict(base); p_['learning_rate']=lr; p_['max_depth']=dp
    mm = xgb.XGBRegressor(n_estimators=ne, tree_method='hist', n_jobs=4, verbosity=0, **p_).fit(Xtr,ytr)
    print(f"ne={ne} lr={lr} depth={dp}: MAE", round(np.abs(mm.predict(Xev)-yev).mean(),3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS], tr.future_spend_4w
Xev, yev = ev[FCOLS], ev.future_spend_4w
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
pev = mdl.predict(Xev)

def ev_mae(newcols_tr, newcols_ev, name):
    Xtr2 = np.column_stack([Xtr, newcols_tr]); Xev2 = np.column_stack([Xev, newcols_ev])
    mm = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr2, ytr)
    print(name, "MAE:", round(np.abs(mm.predict(Xev2)-yev).mean(),4))

# 1. dispersion features: std of the 4 lagged 4w spends
lagcols = ['lag1_spend','lag2_spend','lag3_spend']
for nm, cols in [('lag_std', lagcols), ('lag_std+cv', lagcols+['wk_std8'])]:
    v_tr = tr[cols].std(axis=1).values; v_ev = ev[cols].std(axis=1).values
    ev_mae(v_tr, v_ev, nm)

# 2. blend feature: model prediction as feature (stacking) - use in-sample pred
ev_mae(mdl.predict(Xtr)[:,None], pev[:,None], "stack_pred_as_feat")

# 3. interactions: spend28*tenure, spend28*ntrips
for a,b in [('spend_28','tenure'),('spend_28','trips_28'),('exp4w_blend','tenure')]:
    ev_mae((tr[a]*tr[b]).values[:,None], (ev[a]*ev[b]).values[:,None], f"int_{a}x{b}")

# 4. household-level long-run aggregates: mean/max/std of all past 4w-window spends
# approximate via exp4w_all + spend_all/tenure etc - test adding spend_all/tenure ratio
ev_mae((tr.spend_all/np.maximum(tr.tenure,1)).values[:,None], (ev.spend_all/np.maximum(ev.tenure,1)).values[:,None], "spend_per_day_all")


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
# encode strings
for c in m.columns:
    if m[c].dtype==object: m[c] = m[c].astype('category').cat.codes
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS].values.astype(float), tr.future_spend_4w.values
Xev, yev = ev[FCOLS].values.astype(float), ev.future_spend_4w.values
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
pev = mdl.predict(Xev)
print("refit encoded MAE:", round(np.abs(pev-yev).mean(),4))

def ev_mae(nc_tr, nc_ev, name):
    Xtr2 = np.column_stack([Xtr, np.asarray(nc_tr, dtype=float).reshape(len(tr),-1)])
    Xev2 = np.column_stack([Xev, np.asarray(nc_ev, dtype=float).reshape(len(ev),-1)])
    mm = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr2, ytr)
    print(name, "MAE:", round(np.abs(mm.predict(Xev2)-yev).mean(),4))

ev_mae(tr[lagcols:=['lag1_spend','lag2_spend','lag3_spend']].std(axis=1).values, ev[lagcols].std(axis=1).values, "lag_std")
ev_mae(mdl.predict(Xtr)[:,None], pev[:,None], "stack_pred_as_feat")
for a,b in [('spend_28','tenure'),('spend_28','trips_28'),('exp4w_blend','tenure')]:
    ev_mae((tr[a]*tr[b]).values, (ev[a]*ev[b]).values, f"int_{a}x{b}")
ev_mae((tr.spend_all/np.maximum(tr.tenure,1)).values, (ev.spend_all/np.maximum(ev.tenure,1)).values, "spend_per_day_all")


# ---- cell ----

import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved('feats_v4.parquet')
print(feats.dtypes[feats.dtypes=='object'].index.tolist())
print(feats.dtypes.tail(12))
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
print([ (c, str(m[c].dtype)) for c in m.columns if m[c].dtype not in (np.number,) ][:20])


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day','snap_day','snap_week','snap_cycle_pos']
for c in m.columns:
    if str(m[c].dtype)=='category': m[c] = m[c].cat.codes
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS].values.astype(float), tr.future_spend_4w.values
Xev, yev = ev[FCOLS].values.astype(float), ev.future_spend_4w.values
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
mdl = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr, ytr)
pev = mdl.predict(Xev)
print("refit encoded MAE:", round(np.abs(pev-yev).mean(),4))

def ev_mae(nc_tr, nc_ev, name):
    Xtr2 = np.column_stack([Xtr, np.asarray(nc_tr, dtype=float).reshape(len(tr),-1)])
    Xev2 = np.column_stack([Xev, np.asarray(nc_ev, dtype=float).reshape(len(ev),-1)])
    mm = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **base).fit(Xtr2, ytr)
    print(name, "MAE:", round(np.abs(mm.predict(Xev2)-yev).mean(),4))

lagcols=['lag1_spend','lag2_spend','lag3_spend']
ev_mae(tr[lagcols].std(axis=1).values, ev[lagcols].std(axis=1).values, "lag_std")
ev_mae(mdl.predict(Xtr)[:,None], pev[:,None], "stack_pred_as_feat")
for a,b in [('spend_28','tenure'),('spend_28','trips_28'),('exp4w_blend','tenure')]:
    ev_mae((tr[a]*tr[b]).values, (ev[a]*ev[b]).values, f"int_{a}x{b}")
ev_mae((tr.spend_all/np.maximum(tr.tenure,1)).values, (ev.spend_all/np.maximum(ev.tenure,1)).values, "spend_per_day_all")


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day']
for c in m.columns:
    if str(m[c].dtype)=='category': m[c] = m[c].cat.codes
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS].values.astype(float), tr.future_spend_4w.values
Xev, yev = ev[FCOLS].values.astype(float), ev.future_spend_4w.values
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.5)
def q(**kw):
    p=dict(base); p.update(kw)
    return xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **p)
mdl = q().fit(Xtr, ytr); pev = mdl.predict(Xev)
print("refit WITH snap feats MAE:", round(np.abs(pev-yev).mean(),4))

for a in [0.45,0.55]:
    mm=q(quantile_alpha=a).fit(Xtr,ytr); print(f"alpha={a}:", round(np.abs(mm.predict(Xev)-yev).mean(),4))

# multiplicative residual on exp4w_blend
eb_tr, eb_ev = tr.exp4w_blend.values, ev.exp4w_blend.values
mm = q().fit(Xtr, np.log((ytr+1)/(eb_tr+1)))
pm = np.maximum(eb_ev+1,0)*np.exp(mm.predict(Xev))-1
print("mult-resid-on-blend:", round(np.abs(pm-yev).mean(),4))

# ratio target: y / snapshot global mean
gm = tr.groupby('snapshot_day').future_spend_4w.mean()
mm = q().fit(Xtr, ytr/tr.snapshot_day.map(gm).values)
pm2 = mm.predict(Xev)*ev.snapshot_day.map(gm).values
print("ratio-target:", round(np.abs(pm2-yev).mean(),4))

# HistGB blend
hg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_depth=None, min_samples_leaf=60, loss='absolute_error', random_state=0).fit(Xtr, ytr)
ph = hg.predict(Xev)
print("HistGB MAE:", round(np.abs(ph-yev).mean(),4))
for w in [0.3,0.5]:
    print(f"blend xgb+hist w={w}:", round(np.abs(w*pev+(1-w)*ph-yev).mean(),4))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
DROP = ['household_key','snapshot_day']
for c in m.columns:
    if str(m[c].dtype)=='category': m[c] = m[c].cat.codes
FCOLS = [c for c in feats.columns if c not in DROP]
tr = m[m.snapshot_day<=375]; ev = m[m.snapshot_day>=403]
Xtr, ytr = tr[FCOLS].values.astype(float), tr.future_spend_4w.values
Xev, yev = ev[FCOLS].values.astype(float), ev.future_spend_4w.values
base = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.55)
def q(**kw):
    p=dict(base); p.update(kw)
    return xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **p)

# with interactions
I_tr = np.column_stack([tr.spend_28*tr.tenure, tr.spend_28*tr.trips_28, tr.exp4w_blend*tr.tenure]).astype(float)
I_ev = np.column_stack([ev.spend_28*ev.tenure, ev.spend_28*ev.trips_28, ev.exp4w_blend*ev.tenure]).astype(float)
Xtr2, Xev2 = np.column_stack([Xtr,I_tr]), np.column_stack([Xev,I_ev])
x1 = q().fit(Xtr2, ytr); p1 = x1.predict(Xev2)
hg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, min_samples_leaf=60, loss='absolute_error', random_state=0).fit(Xtr2, ytr)
p2 = hg.predict(Xev2)
print("combo alpha55+int, xgb:", round(np.abs(p1-yev).mean(),4))
print("combo blend 0.7/0.3:", round(np.abs(0.7*p1+0.3*p2-yev).mean(),4))
print("combo blend 0.6/0.4:", round(np.abs(0.6*p1+0.4*p2-yev).mean(),4))
print("combo blend 0.5/0.5:", round(np.abs(0.5*p1+0.5*p2-yev).mean(),4))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
m = tt.merge(feats, on=['household_key','snapshot_day'])
FCOLS = [c for c in feats.columns if c not in ['household_key','snapshot_day']]
def enc(df):
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype)=='category': df[c] = df[c].cat.codes
    return df
Xtr = enc(m[FCOLS]).values.astype(float); ytr = m.future_spend_4w.values
val = feats[feats.snapshot_day>=459].sort_values(['household_key','snapshot_day'])
Xva = enc(val[FCOLS]).values.astype(float)

xgb_params = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, subsample=0.9,
                  colsample_bytree=0.8, reg_lambda=1.0, objective='reg:quantileerror', quantile_alpha=0.55)
mx = xgb.XGBRegressor(n_estimators=400, tree_method='hist', n_jobs=4, verbosity=0, **xgb_params).fit(Xtr, ytr)
px = mx.predict(Xva)
mh = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, min_samples_leaf=60,
                                   loss='absolute_error', random_state=0).fit(Xtr, ytr)
ph = mh.predict(Xva)
pred = 0.7*px + 0.3*ph
out = pd.DataFrame({'household_key': val.household_key.values,
                    'snapshot_day': val.snapshot_day.values,
                    'prediction': np.clip(pred, 0, None)})
print(out.shape, out.prediction.describe())
p = agent_api.save_table(out, 'pred_e012.parquet')
print(p)
