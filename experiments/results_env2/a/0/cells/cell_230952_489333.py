
import pandas as pd, numpy as np, agent_api, xgboost as xgb
from sklearn.isotonic import IsotonicRegression

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0,obj='reg:quantileerror'):
    kw = dict(objective=obj, max_depth=depth, min_child_weight=mcw, n_estimators=nest,
              learning_rate=lr, subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    if obj=='reg:quantileerror': kw['quantile_alpha']=alpha
    m = xgb.XGBRegressor(**kw); m.fit(Xtr,ytr); return m

tr375 = df[df.snapshot_day<=375]; tr403 = df[df.snapshot_day<=403]
cal403 = df[df.snapshot_day==403]; va431 = df[df.snapshot_day==431]
X375,y375 = make_xy(tr375); Xc,yc = make_xy(cal403); Xv,yv = make_xy(va431)
blv = Xv['exp4w_blend'].values; blc = Xc['exp4w_blend'].values

# honest calibration test: train<=375, calibrate on 403, evaluate on 431
m = qmodel(X375,y375)
pc = m.predict(Xc); pv = m.predict(Xv)
iso = IsotonicRegression(out_of_bounds='clip', y_min=0)
iso.fit(pc, yc)
pv_iso = iso.predict(pv)
print("honest iso-calib: raw MAE(431):", np.abs(np.clip(0.7*pv+0.3*blv,0,None)-yv).mean(),
      " iso MAE:", np.abs(np.clip(0.7*pv_iso+0.3*blv,0,None)-yv).mean())
# linear recalibration on cal set
a,b = np.polyfit(pc,yc,1)
print("linear recal a,b:",round(a,3),round(b,1), " MAE:", np.abs(np.clip(0.7*(a*pv+b)+0.3*blv,0,None)-yv).mean())

# --- OOF top-end boost: fit g on train OOF (train<=403, eval 431)
tr = df[df.snapshot_day<=403]; Xtr,ytr = make_xy(tr)
# OOF by snapshot
oof = np.zeros(len(tr))
for s in sorted(tr.snapshot_day.unique()):
    itr = tr[tr.snapshot_day!=s]; it = tr[tr.snapshot_day==s]
    Xi,yi = make_xy(itr); Xt,yt = make_xy(it)
    mm = qmodel(Xi,yi); oof[tr.index.get_indexer(it.index)] = mm.predict(Xt)
oof = np.clip(0.7*oof+0.3*tr['exp4w_blend'].values,0,None)
yo = tr['future_spend_4w'].values
# top-end boost: p' = p + g*max(0,p-T)
for T in [200,300,400]:
    x = np.maximum(0,oof-T)
    g = np.clip(np.polyfit(x, yo-oof, 1)[0], 0, 0.5)
    pv2 = np.clip(0.7*pv+0.3*blv,0,None)
    print(f"top-end boost T={T}: g={g:.3f}, MAE:", np.abs((pv2+g*np.maximum(0,pv2-T))-yv).mean())

# --- ensembles
m1 = qmodel(Xtr,ytr,seed=1); m2 = qmodel(Xtr,ytr,seed=2,mcw=10)
m3 = qmodel(Xtr,ytr,depth=5,mcw=15,nest=800,seed=3)
m4 = qmodel(Xtr,ytr,obj='reg:pseudohubererror',nest=600,seed=4)
m5 = qmodel(Xtr,ytr,obj='reg:squarederror',nest=600,seed=5)
P = {}
for nm,mm in [('s1',m1),('s2',m2),('d5',m3),('hub',m4),('sq',m5)]:
    P[nm] = np.clip(0.7*mm.predict(Xv)+0.3*blv,0,None)
    print(f"{nm}: MAE {np.abs(P[nm]-yv).mean():.3f}")
ens = (P['s1']+P['s2']+P['d5'])/3
print("ens(s1,s2,d5):", np.abs(ens-yv).mean())
ens2 = (P['s1']+P['s2']+P['d5']+P['hub'])/4
print("ens+hub:", np.abs(ens2-yv).mean())
ens3 = 0.5*P['s1']+0.25*P['d5']+0.25*P['sq']
print("ens q+sq:", np.abs(ens3-yv).mean())
