
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
