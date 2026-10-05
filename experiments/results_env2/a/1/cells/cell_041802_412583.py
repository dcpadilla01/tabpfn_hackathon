import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
tt = api.train_targets()[['household_key','snapshot_day','future_spend_4w']]
df = allF.drop(columns=['future_spend_4w']).merge(tt, on=['household_key','snapshot_day'])
BASE = [c for c in allF.columns if c not in ['household_key','snapshot_day','future_spend_4w']]
tr = df[df.snapshot_day<=375]; te = df[df.snapshot_day.isin([403,431])]
ytr = tr['future_spend_4w'].values; yte = te['future_spend_4w'].values

def run(FEATS, lr=0.03, rounds=1200, hl=140, depth=6, mcw=25, ss=0.7, cs=0.7, seeds=(1,2), ref=375, tag=""):
    t0=time.time(); preds=[]
    wt = 0.5**((ref-tr['snapshot_day'].values)/hl)
    for s in seeds:
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=rounds, learning_rate=lr,
                             max_depth=depth, min_child_weight=mcw, subsample=ss, colsample_bytree=cs, n_jobs=8, random_state=s, tree_method='hist')
        m.fit(tr[FEATS], ytr, sample_weight=wt)
        preds.append(np.clip(m.predict(te[FEATS]),0,None))
    P = np.mean(preds,axis=0)
    mae = np.abs(P-yte).mean()
    print(f"{tag:34s} honest MAE={mae:.3f} bias={(P-yte).mean():6.1f} ({time.time()-t0:.0f}s)")
    return P, mae

_, m0 = run(BASE, tag="base (E011 recipe)")
F2 = BASE + ['snapshot_day']
_, m1 = run(F2, tag="+snapshot_day feature")
_, m2 = run(BASE, rounds=2400, lr=0.02, tag="2400r lr.02")
_, m3 = run(BASE, hl=277, tag="hl=277")
_, m4 = run(BASE, hl=70, tag="hl=70")
_, m5 = run(BASE, hl=100000, tag="no decay")
_, m6 = run(BASE, depth=5, tag="depth5")
_, m7 = run(BASE, depth=8, mcw=50, tag="depth8 mcw50")
_, m8 = run(BASE, ss=0.9, cs=0.9, tag="ss.9 cs.9")
print("\nbest so far:", min([(m0,'base'),(m1,'+day'),(m2,'2400r'),(m3,'hl277'),(m4,'hl70'),(m5,'nodecay'),(m6,'d5'),(m7,'d8'),(m8,'ss9')]))
