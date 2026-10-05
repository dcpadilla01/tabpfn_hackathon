import pandas as pd, numpy as np

tt = agent_api.train_targets()
print('targets shape', tt.shape)
print(tt['future_spend_4w'].describe())
print('zero frac %.4f' % (tt['future_spend_4w']==0).mean())
print(tt['future_spend_4w'].quantile([0,.05,.1,.25,.5,.75,.9,.95,.99,1]).to_dict())
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median','std'])
g['zerofrac'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda s:(s==0).mean())
print(g)

for name in ['allF','repro_e5','e005_preds','e005_newfeats','e004_features','e004_new','lagfeats','lagfeats2','f_weekly','camp_feats']:
    try:
        df = agent_api.load_saved(name+'.parquet')
        print('===', name, df.shape)
        cols = list(df.columns)
        print(cols[:55])
        if 'snapshot_day' in cols:
            print('snap days:', sorted(df['snapshot_day'].unique().tolist()))
    except Exception as e:
        print(name, 'ERR', repr(e))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)

allF = agent_api.load_saved('allF.parquet')
print('allF cols (%d):' % allF.shape[1])
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day')]
print(feat_cols)
print(allF[feat_cols].dtypes.value_counts())
# NaN structure by snapshot for a few cols
sub = allF.groupby('snapshot_day')[['spend_364','spend_lag1y','spend_seas_364','seas_ok','sp28_yag','wk_52']].apply(lambda d: d.isna().mean().round(3))
print(sub)

tt = agent_api.train_targets()
allF2 = allF.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = allF2[allF2.snapshot_day<=431]
print('\nmean target vs mean spend_28 by snapshot (trend check):')
g = tr.groupby('snapshot_day').agg(y=('future_spend_4w','mean'), s28=('spend_28','mean'), s84=('spend_84','mean'), n=('future_spend_4w','size'))
g['y_over_s28'] = g.y/g.s28
print(g.round(3))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print(len(feat_cols))
sub = allF.groupby('snapshot_day')[['spend_364','spend_lag1y','spend_seas_364','seas_ok','s1','s2','s3','s4','wk_13','wk_26']].apply(lambda d: d.isna().mean().round(3))
print(sub)
tt = agent_api.train_targets()
allF2 = allF.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = allF2[allF2.snapshot_day<=431]
g = tr.groupby('snapshot_day').agg(y=('future_spend_4w','mean'), s28=('spend_28','mean'), s84=('spend_84','mean'), n=('future_spend_4w','size'))
g['y_over_s28'] = g.y/g.s28
print(g.round(3))
# distribution of y and spend_28 overall
print('\nspend_28 describe'); print(tr['spend_28'].describe())
print('\nzero28 frac', tr['zero28'].mean())
# how well does spend_28 alone predict y?
from sklearn.metrics import mean_absolute_error
print('MAE spend_28 as pred:', mean_absolute_error(tr['future_spend_4w'], tr['spend_28']))
print('MAE spend_84*0.25... spend_84/3:', mean_absolute_error(tr['future_spend_4w'], tr['spend_84']/3))
print('MAE spend_364/13:', mean_absolute_error(tr['future_spend_4w'], tr['spend_364']/13))
print('MAE max(spend_28, s84/3):', mean_absolute_error(tr['future_spend_4w'], np.maximum(tr['spend_28'], tr['spend_84']/3)))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print(len(feat_cols))
sub = allF.groupby('snapshot_day')[['spend_364','spend_lag1y','spend_seas_364','seas_ok','s1','s2','s3','s4']].apply(lambda d: d.isna().mean().round(3))
print(sub)
tt = agent_api.train_targets()
allF2 = allF.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = allF2[allF2.snapshot_day<=431]
g = tr.groupby('snapshot_day').agg(y=('future_spend_4w','mean'), s28=('spend_28','mean'), s84=('spend_84','mean'), n=('future_spend_4w','size'))
g['y_over_s28'] = g.y/g.s28
print(g.round(3))
from sklearn.metrics import mean_absolute_error
print('MAE spend_28 as pred:', mean_absolute_error(tr['future_spend_4w'], tr['spend_28']))
print('MAE spend_84/3:', mean_absolute_error(tr['future_spend_4w'], tr['spend_84']/3))
print('MAE spend_364/13:', mean_absolute_error(tr['future_spend_4w'], tr['spend_364']/13))
print('MAE max(spend_28, s84/3):', mean_absolute_error(tr['future_spend_4w'], np.maximum(tr['spend_28'], tr['spend_84']/3)))
print('MAE blend .5:', mean_absolute_error(tr['future_spend_4w'], 0.5*tr['spend_28']+0.5*tr['spend_84']/3))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
g = tr.groupby('snapshot_day').agg(y=('future_spend_4w','mean'), s28=('spend_28','mean'), s84=('spend_84','mean'), n=('future_spend_4w','size'))
g['y_over_s28'] = g.y/g.s28
print(g.round(3))
from sklearn.metrics import mean_absolute_error
y = tr['future_spend_4w']
print('MAE spend_28 as pred:', mean_absolute_error(y, tr['spend_28']))
print('MAE spend_84/3:', mean_absolute_error(y, tr['spend_84']/3))
print('MAE spend_364/13:', mean_absolute_error(y, tr['spend_364']/13))
print('MAE max(spend_28, s84/3):', mean_absolute_error(y, np.maximum(tr['spend_28'], tr['spend_84']/3)))
print('MAE blend .5:', mean_absolute_error(y, 0.5*tr['spend_28']+0.5*tr['spend_84']/3))
# per-snapshot MAE of spend_28
print('\nper-snapshot MAE of spend_28:')
print(tr.groupby('snapshot_day').apply(lambda d: mean_absolute_error(d['future_spend_4w'], d['spend_28']), include_groups=False).round(2))
# check NaN of s1..s4 etc at late snapshots
sub = allF.groupby('snapshot_day')[['s1','s2','s3','s4','seq_mean','ratio_s1_s3','gap_mean','wk_inact_streak','wk_best_streak','dsp28_GROC']].apply(lambda d: d.isna().mean().round(3))
print('\nNaN rates late snapshots:'); print(sub.loc[[403,431,459,487,515,543]])

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
y = tr['future_spend_4w']
from sklearn.metrics import mean_absolute_error
print('MAE spend_364/13:', mean_absolute_error(y, tr['spend_364'].fillna(0)/13))
print('MAE max(spend_28, s84/3):', mean_absolute_error(y, np.maximum(tr['spend_28'], tr['spend_84'].fillna(0)/3)))
print('MAE blend .5:', mean_absolute_error(y, 0.5*tr['spend_28']+0.5*tr['spend_84'].fillna(0)/3))
print('\nper-snapshot MAE of spend_28:')
print(tr.groupby('snapshot_day').apply(lambda d: mean_absolute_error(d['future_spend_4w'], d['spend_28']), include_groups=False).round(2))
# NaN rates of new features at late snapshots
sub = allF.groupby('snapshot_day')[['s1','s2','s3','s4','seq_mean','ratio_s1_s3','gap_mean','wk_inact_streak','wk_best_streak','dsp28_GROC','drec_GROC','dept28_tot']].apply(lambda d: d.isna().mean().round(3))
print('\nNaN rates:'); print(sub.loc[[95,431,459,487,515,543]])
# how many rows per snapshot in allF
print('\nrows per snapshot:'); print(allF.snapshot_day.value_counts().sort_index())
# check e005 preds per snapshot MAE
e5 = agent_api.load_saved('e005_preds.parquet')
tt = agent_api.train_targets()
val = tt[tt.snapshot_day>431].merge(e5, on=['household_key','snapshot_day'])
print('\ne005 per-snapshot MAE:')
print(val.groupby('snapshot_day').apply(lambda d: mean_absolute_error(d['future_spend_4w'], d['prediction']), include_groups=False).round(2))
print('overall', mean_absolute_error(val['future_spend_4w'], val['prediction']).round(3))
# bias
d = val['prediction']-val['future_spend_4w']
print('mean bias', d.mean().round(3), 'median bias', d.median().round(3))
qb = pd.qcut(val['future_spend_4w'], 10, duplicates='drop')
print(val.groupby(qb, observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g['future_spend_4w'].mean(),'p':g['prediction'].mean(),'bias':(g['prediction']-g['future_spend_4w']).mean()}), include_groups=False).round(2))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
tt = agent_api.train_targets()
e5 = agent_api.load_saved('e005_preds.parquet')
print(tt.dtypes, '\n', e5.dtypes)
print(tt['snapshot_day'].unique()[:5], e5['snapshot_day'].unique()[:5])
print(tt['household_key'].dtype, e5['household_key'].dtype, tt['household_key'].iloc[0], e5['household_key'].iloc[0])
m = tt.merge(e5, on=['household_key','snapshot_day'])
print('merged', m.shape)
from sklearn.metrics import mean_absolute_error
print('MAE e005:', mean_absolute_error(m['future_spend_4w'], m['prediction']))
print(m.groupby('snapshot_day').apply(lambda d: mean_absolute_error(d['future_spend_4w'], d['prediction']), include_groups=False).round(2))
d = m['prediction']-m['future_spend_4w']
print('mean bias', d.mean().round(3), 'median bias', d.median().round(3))
qb = pd.qcut(m['future_spend_4w'], 10, duplicates='drop')
print(m.groupby(qb, observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g['future_spend_4w'].mean(),'p':g['prediction'].mean(),'bias':(g['prediction']-g['future_spend_4w']).mean()}), include_groups=False).round(2))
# check repro_e5 vs e005_preds
r5 = agent_api.load_saved('repro_e5.parquet')
mm = e5.merge(r5, on=['household_key','snapshot_day'])
print('repro check: corr p13 vs prediction', np.corrcoef(mm['p13'], mm['prediction'])[0,1].round(4), 'mean abs diff', (mm['p13']-mm['prediction']).abs().mean().round(3))
print('p11 vs p13 corr', np.corrcoef(mm['p11'], mm['p13'])[0,1].round(4), 'mean abs diff', (mm['p11']-mm['p13']).abs().mean().round(3))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
tt = agent_api.train_targets()
# NOTE: train_targets only covers train snapshots
print('train_targets snapshots:', sorted(tt.snapshot_day.unique()))
tr = allF[allF.snapshot_day<=431].copy()
y = tr['future_spend_4w']
from sklearn.metrics import mean_absolute_error as mae

# internal holdout: fit on snapshots <=375, evaluate on 403+431
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
print('fit rows', len(fit), 'holdout rows', len(ho))

def rep(name, yt, yp):
    print('%-28s MAE %.3f  bias %+.2f' % (name, np.abs(yt-yp).mean(), (yp-yt).mean()))

# 1) simple baselines on holdout
rep('spend_28', ho.future_spend_4w, ho.spend_28)
rep('spend_84/3', ho.future_spend_4w, ho.spend_84/3)
rep('spend_56/2', ho.future_spend_4w, ho.spend_56/2)
rep('blend .5', ho.future_spend_4w, .5*ho.spend_28+.5*ho.spend_84/3)
rep('blend .35', ho.future_spend_4w, .35*ho.spend_28+.65*ho.spend_84/3)
rep('spend_112/4', ho.future_spend_4w, ho.spend_112/4)
rep('spend_168/6', ho.future_spend_4w, ho.spend_168/6)

# 2) linear calibration of spend_28 fitted on fit-set
b = np.polyfit(fit.spend_28, fit.future_spend_4w, 1)
print('linear fit slope/intercept', b.round(4))
rep('lin(spend_28)', ho.future_spend_4w, np.clip(np.polyval(b, ho.spend_28), 0, None))
# isotonic
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip').fit(fit.spend_28, fit.future_spend_4w)
rep('iso(spend_28)', ho.future_spend_4w, iso.predict(ho.spend_28))
# quantile-ish: median regression via isotonic on quantile
iso5 = IsotonicRegression(out_of_bounds='clip').fit(fit.spend_28, fit.future_spend_4w)
# 3) two-feature OLS: spend_28, spend_84/3
X = np.column_stack([fit.spend_28, fit.spend_84/3, np.ones(len(fit))])
coef, *_ = np.linalg.lstsq(X, fit.future_spend_4w, rcond=None)
print('ols coef', coef.round(4))
Xh = np.column_stack([ho.spend_28, ho.spend_84/3, np.ones(len(ho))])
rep('ols2', ho.future_spend_4w, np.clip(Xh@coef,0,None))
# 4) gradient boosting light on 2 feats
import xgboost as xgb
m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, objective='reg:absoluteerror', n_jobs=8)
m.fit(fit[['spend_28','spend_84','spend_364','actdays_28','actdays_84','recency','tenure']].fillna(0), fit.future_spend_4w)
rep('xgb-abs small', ho.future_spend_4w, m.predict(ho[['spend_28','spend_84','spend_364','actdays_28','actdays_84','recency','tenure']].fillna(0)))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]

def mae(yt, yp): return np.abs(yt-yp).mean()

# XGB quantile (alpha=.5) like E005 on all features; internal holdout to measure variance
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def fit_xgb(Xtr, ytr, Xho, seeds=(7,), rounds=1200, lr=0.03, decay=140):
    t0 = (Xtr.snapshot_day.max() if 'snapshot_day' in Xtr else 375)
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)
    preds = []
    for s in seeds:
        m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=lr, max_depth=6, min_child_weight=10,
                             subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, reg_alpha=0.0,
                             objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist', n_jobs=8, random_state=s)
        m.fit(Xtr[feat_cols].values, ytr, sample_weight=w)
        preds.append(m.predict(Xho[feat_cols].values))
    return np.mean(preds, axis=0)

Xtr, ytr = fit, fit.future_spend_4w.values
p = fit_xgb(Xtr, ytr, ho, seeds=(7,11,13,19), rounds=1200)
print('internal holdout MAE (E005-style, 4 seeds):', mae(ho.future_spend_4w, p).round(3))
print('bias', (p-ho.future_spend_4w).mean().round(2))
# per-snapshot
hh = ho.copy(); hh['p']=p
print(hh.groupby('snapshot_day').apply(lambda d: mae(d.future_spend_4w, d.p), include_groups=False).round(2))
# compare spend_84/3 on same holdout
print('spend_84/3 MAE:', mae(ho.future_spend_4w, ho.spend_84/3).round(3))
print('blend .35 MAE:', mae(ho.future_spend_4w, .35*ho.spend_28+.65*ho.spend_84/3).round(3))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
import xgboost as xgb

def train_pred(Xtr, ytr, Xho, seeds, rounds=1200, lr=0.03, decay=140, depth=6, mcw=10):
    t0 = Xtr.snapshot_day.max()
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)
    ps=[]
    for s in seeds:
        m = xgb.XGBRegressor(n_estimators=rounds, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                             subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
                             objective='reg:quantileerror', quantile_alpha=0.5, tree_method='hist', n_jobs=8, random_state=s)
        m.fit(Xtr[feat_cols].values, ytr, sample_weight=w)
        ps.append(m.predict(Xho[feat_cols].values))
    return np.mean(ps, axis=0)

def mae(yt, yp): return np.abs(yt-yp).mean()

# A) quantile on log1p(y)
p_log = train_pred(fit, np.log1p(fit.future_spend_4w.values), ho)
print('A log-target quantile :', mae(ho.future_spend_4w, np.expm1(p_log)).round(3))

# B) absoluteerror objective
t0 = fit.snapshot_day.max()
w = np.power(0.5, (t0-fit.snapshot_day.values)/140)
m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=1.0, objective='reg:absoluteerror', tree_method='hist', n_jobs=8, random_state=7)
m.fit(fit[feat_cols].values, fit.future_spend_4w.values, sample_weight=w)
p_abs = m.predict(ho[feat_cols].values)
print('B absoluteerror       :', mae(ho.future_spend_4w, p_abs).round(3))

# C) squared error
m2 = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=1.0, objective='reg:squarederror', tree_method='hist', n_jobs=8, random_state=7)
m2.fit(fit[feat_cols].values, fit.future_spend_4w.values, sample_weight=w)
p_sq = m2.predict(ho[feat_cols].values)
print('C squarederror        :', mae(ho.future_spend_4w, p_sq).round(3))
print('  bias C', (p_sq-ho.future_spend_4w).mean().round(2), ' bias A', (np.expm1(p_log)-ho.future_spend_4w).mean().round(2))

# D) blend A+C
print('D blend .5A+.5C       :', mae(ho.future_spend_4w, .5*np.expm1(p_log)+.5*p_sq).round(3))
print('D blend .7C+.3A       :', mae(ho.future_spend_4w, .7*p_sq+.3*np.expm1(p_log)).round(3))
print('E blend C + spend84/3 :', mae(ho.future_spend_4w, .8*p_sq+.2*ho.spend_84/3).round(3))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
import xgboost as xgb

def train_pred(Xtr, ytr, Xho, seeds=(7,), rounds=1200, lr=0.03, decay=140, depth=6, mcw=10, obj='reg:quantileerror'):
    t0 = Xtr.snapshot_day.max()
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)
    ps=[]
    for s in seeds:
        kw = dict(n_estimators=rounds, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method='hist', n_jobs=8, random_state=s)
        if obj=='reg:quantileerror': kw.update(objective=obj, quantile_alpha=0.5)
        else: kw['objective']=obj
        m = xgb.XGBRegressor(**kw)
        m.fit(Xtr[feat_cols].values, ytr, sample_weight=w)
        ps.append(m.predict(Xho[feat_cols].values))
    return np.mean(ps, axis=0)

def mae(yt, yp): return np.abs(yt-yp).mean()

p_log = train_pred(fit, np.log1p(fit.future_spend_4w.values), ho)
print('A log-target quantile :', mae(ho.future_spend_4w, np.expm1(p_log)).round(3))

t0 = fit.snapshot_day.max()
w = np.power(0.5, (t0-fit.snapshot_day.values)/140)
m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=1.0, objective='reg:absoluteerror', tree_method='hist', n_jobs=8, random_state=7)
m.fit(fit[feat_cols].values, fit.future_spend_4w.values, sample_weight=w)
p_abs = m.predict(ho[feat_cols].values)
print('B absoluteerror       :', mae(ho.future_spend_4w, p_abs).round(3))

m2 = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=1.0, objective='reg:squarederror', tree_method='hist', n_jobs=8, random_state=7)
m2.fit(fit[feat_cols].values, fit.future_spend_4w.values, sample_weight=w)
p_sq = m2.predict(ho[feat_cols].values)
print('C squarederror        :', mae(ho.future_spend_4w, p_sq).round(3))
print('  bias A', (np.expm1(p_log)-ho.future_spend_4w).mean().round(2), ' bias B', (p_abs-ho.future_spend_4w).mean().round(2), ' bias C', (p_sq-ho.future_spend_4w).mean().round(2))
print('D blend .5A+.5C       :', mae(ho.future_spend_4w, .5*np.expm1(p_log)+.5*p_sq).round(3))
print('D blend .7C+.3A       :', mae(ho.future_spend_4w, .7*p_sq+.3*np.expm1(p_log)).round(3))
print('D blend .3B+.7C       :', mae(ho.future_spend_4w, .3*p_abs+.7*p_sq).round(3))
print('E blend C + spend84/3 :', mae(ho.future_spend_4w, .8*p_sq+.2*ho.spend_84/3).round(3))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
import xgboost as xgb

def train_pred(Xtr, ytr, Xho, seeds=(7,), rounds=1200, lr=0.03, decay=140, depth=6, mcw=10, obj='reg:quantileerror', wmul=1.0):
    t0 = Xtr.snapshot_day.max()
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)*wmul
    ps=[]
    for s in seeds:
        kw = dict(n_estimators=rounds, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method='hist', n_jobs=8, random_state=s)
        if obj=='reg:quantileerror': kw.update(objective=obj, quantile_alpha=0.5)
        else: kw['objective']=obj
        m = xgb.XGBRegressor(**kw)
        m.fit(Xtr[feat_cols].values, ytr, sample_weight=w)
        ps.append(m.predict(Xho[feat_cols].values))
    return np.mean(ps, axis=0)

def mae(yt, yp): return np.abs(yt-yp).mean()

p_log = np.expm1(train_pred(fit, np.log1p(fit.future_spend_4w.values), ho))
# clip log predictions at max of train targets? check
print('log pred max', p_log.max().round(1), 'train y max', fit.future_spend_4w.max())

# clip variants
for cap in [1500, 1200, 1000, 900]:
    print('clip@%d:'%cap, mae(ho.future_spend_4w, np.clip(p_log,0,cap)).round(3))

# per-decile bias of log model
qb = pd.qcut(ho.future_spend_4w, 10, duplicates='drop')
g = ho.assign(p=p_log).groupby(qb, observed=True).apply(lambda d: pd.Series({'n':len(d),'y':d.future_spend_4w.mean(),'p':d.p.mean()}), include_groups=False)
print(g.round(2))
# what if we add a positive shift? (bias -22)
for sh in [0, 5, 10, 15]:
    print('shift+%d:'%sh, mae(ho.future_spend_4w, p_log+sh).round(3))
# multiplicative calibration
for k in [1.0, 1.03, 1.05, 1.08]:
    print('x%.2f:'%k, mae(ho.future_spend_4w, p_log*k).round(3))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
import xgboost as xgb

def train_pred(Xtr, ytr, Xho, seeds=(7,), rounds=1200, lr=0.03, decay=140, depth=6, mcw=10, obj='reg:quantileerror'):
    t0 = Xtr.snapshot_day.max()
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)
    ps=[]
    for s in seeds:
        kw = dict(n_estimators=rounds, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method='hist', n_jobs=8, random_state=s)
        if obj=='reg:quantileerror': kw.update(objective=obj, quantile_alpha=0.5)
        else: kw['objective']=obj
        m = xgb.XGBRegressor(**kw)
        m.fit(Xtr[feat_cols].values, ytr, sample_weight=w)
        ps.append(m.predict(Xho[feat_cols].values))
    return np.mean(ps, axis=0)

def mae(yt, yp): return np.abs(yt-yp).mean()

p_log = np.expm1(train_pred(fit, np.log1p(fit.future_spend_4w.values), ho))
# two-model blend grid on internal holdout
p_sq = None
t0 = fit.snapshot_day.max(); w = np.power(0.5,(t0-fit.snapshot_day.values)/140)
m2 = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=1.0, objective='reg:squarederror', tree_method='hist', n_jobs=8, random_state=7)
m2.fit(fit[feat_cols].values, fit.future_spend_4w.values, sample_weight=w)
p_sq = m2.predict(ho[feat_cols].values)
best=(None,9e9)
for wa in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]:
    p = wa*p_log + (1-wa)*p_sq
    v = mae(ho.future_spend_4w, p)
    if v<best[1]: best=(wa,v)
    print('wa=%.1f  %.3f'%(wa,v))
print('best wa', best)
# also blend with spend_84/3
for w3 in [0,.1,.2,.3]:
    p = .7*p_log+.3*p_sq
    p = (1-w3)*p + w3*(ho.spend_84/3)
    print('w3=%.1f'%w3, mae(ho.future_spend_4w,p).round(3))

# ---- cell ----
import pandas as pd, numpy as np
pd.set_option('display.width', 250)
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=431].copy()
fit = tr[tr.snapshot_day<=375]
ho  = tr[tr.snapshot_day>=403]
import xgboost as xgb

def train_pred(Xtr, ytr, Xho, seeds=(7,), rounds=1200, lr=0.03, decay=140, depth=6, mcw=10, obj='reg:quantileerror'):
    t0 = Xtr.snapshot_day.max()
    w = np.power(0.5, (t0 - Xtr.snapshot_day.values)/decay)
    ps=[]
    for s in seeds:
        kw = dict(n_estimators=rounds, learning_rate=lr, max_depth=6, min_child_weight=mcw,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method='hist', n_jobs=8, random_state=s)
        if obj=='reg:quantileerror': kw.update(objective=obj, quantile_alpha=0.5)
        else: kw['objective']=obj
        m = xgb.XGBRegressor(**kw)
        m.fit(Xtr[feat_cols].values, ytr, sample_weight=w)
        ps.append(m.predict(Xho[feat_cols].values))
    return np.mean(ps, axis=0)

def mae(yt, yp): return np.abs(yt-yp).mean()

p_log = np.expm1(train_pred(fit, np.log1p(fit.future_spend_4w.values), ho, seeds=(7,11,13,19)))
t0 = fit.snapshot_day.max(); w = np.power(0.5,(t0-fit.snapshot_day.values)/140)
m2 = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=10, subsample=0.8,
                     colsample_bytree=0.6, reg_lambda=1.0, objective='reg:squarederror', tree_method='hist', n_jobs=8, random_state=7)
m2.fit(fit[feat_cols].values, fit.future_spend_4w.values, sample_weight=w)
p_sq = m2.predict(ho[feat_cols].values)

# grid: wa (log weight), w3 (spend84/3 weight)
best=(None,9e9)
for wa in [0.5,0.6,0.7,0.8]:
    for w3 in [0.0,0.1,0.2,0.3,0.4]:
        p = wa*p_log + (1-wa)*p_sq
        p = (1-w3)*p + w3*(ho.spend_84/3)
        v = mae(ho.future_spend_4w, p)
        if v<best[1]: best=((wa,w3),v)
        print('wa=%.1f w3=%.1f  %.3f'%(wa,w3,v))
print('BEST', best)
# baseline check on same holdout: spend84/3 alone
print('spend84/3 alone:', mae(ho.future_spend_4w, ho.spend_84/3).round(3))
print('blend .35 s28:', mae(ho.future_spend_4w, .35*ho.spend_28+.65*ho.spend_84/3).round(3))

# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time
allF = agent_api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = allF[allF.snapshot_day<=375].copy()          # 11 train snapshots, E005 regime
va = allF[allF.snapshot_day>431].copy()           # validation rows
print('train rows', len(tr), 'val rows', len(va))

t0 = 375
w = np.power(0.5, (t0 - tr.snapshot_day.values)/140.0)

def fit_mean(obj, yvals, seeds, rounds=1200):
    ps=[]
    for s in seeds:
        kw = dict(n_estimators=rounds, learning_rate=0.03, max_depth=6, min_child_weight=10,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method='hist', n_jobs=8, random_state=s)
        if obj=='reg:quantileerror': kw.update(objective=obj, quantile_alpha=0.5)
        else: kw['objective']=obj
        m = xgb.XGBRegressor(**kw)
        m.fit(tr[feat_cols].values, yvals, sample_weight=w)
        ps.append(m.predict(va[feat_cols].values))
    return np.mean(ps, axis=0)

t=time.time()
ylog = np.log1p(tr.future_spend_4w.values)
p_log = np.expm1(fit_mean('reg:quantileerror', ylog, seeds=(7,11,13,19,23,29,31,37)))
print('A done %.0fs'%(time.time()-t))
p_sq  = fit_mean('reg:squarederror', tr.future_spend_4w.values, seeds=(7,11,13))
print('B done %.0fs'%(time.time()-t))

blend = 0.8*p_log + 0.2*p_sq
anchor = va['spend_84'].fillna(0).values/3.0
pred = np.clip(0.7*blend + 0.3*anchor, 0, None)
print('pred stats', np.round([pred.min(), pred.mean(), pred.median(), pred.max()],2))

out = va[['household_key','snapshot_day']].copy()
out['prediction'] = pred
path = agent_api.save_table(out, 'e013_preds.parquet')
print(path, out.shape)