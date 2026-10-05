import pandas as pd, numpy as np, agent_api, xgboost as xgb, time
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
BASE = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4)
DAYS = [375, 403, 431]
act = {d: df[df.snapshot_day==d].future_spend_4w.values for d in DAYS}

def run(train_max, eval_day):
    tr = df[df.snapshot_day <= train_max]
    nz = tr[tr.future_spend_4w > 0]
    mq = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); mq.fit(tr[FE], tr.future_spend_4w)
    mc = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=5,
                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4,
                           eval_metric="logloss")
    mc.fit(tr[FE], (tr.future_spend_4w == 0).astype(int))
    mn = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); mn.fit(nz[FE], nz.future_spend_4w)
    ml = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **BASE); ml.fit(tr[FE], np.log1p(tr.future_spend_4w))
    X = df[df.snapshot_day==eval_day]
    pq = np.clip(mq.predict(X[FE]),0,None)
    pz = mc.predict_proba(X[FE])[:,1]
    pn = np.clip(mn.predict(X[FE]),0,None)
    pl = np.clip(np.expm1(ml.predict(X[FE])),0,None)
    return dict(q50=pq, pz=pz, nz=pn, twopart=(1-pz)*pn, twothr=np.where(pz>0.5,0,pn), log=pl)

t0=time.time()
R = {d: run(d-28, d) for d in DAYS}
print(f"fits done {time.time()-t0:.0f}s")
def ev(key, label):
    maes=[mean_absolute_error(act[d], R[d][key]) for d in DAYS]
    print(f"{label:26s} {[f'{m:.1f}' for m in maes]} avg {np.mean(maes):.3f}")
ev("q50","q50"); ev("log","log-target q50"); ev("nz","nz-only q50")
ev("twopart","twopart (1-pz)*nz"); ev("twothr","twopart thr .5")
for k in [0.5, 0.8]:
    for d in DAYS: R[d][f"nz{k}"] = (1-k*R[d]["pz"])*R[d]["nz"]
    ev(f"nz{k}", f"nz*(1-{k}*pz)")
for w in [0.3, 0.5]:
    for d in DAYS: R[d][f"bl{w}"] = w*R[d]["q50"] + (1-w)*R[d]["twothr"]
    ev(f"bl{w}", f"blend q50+twothr w={w}")
