
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
