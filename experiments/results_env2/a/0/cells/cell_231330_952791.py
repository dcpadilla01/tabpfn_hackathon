
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
