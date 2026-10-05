
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
