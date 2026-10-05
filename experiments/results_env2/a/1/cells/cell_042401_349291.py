import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
tt = api.train_targets()[['household_key','snapshot_day','future_spend_4w']]
df = allF.drop(columns=['future_spend_4w']).merge(tt, on=['household_key','snapshot_day'])
BASE = [c for c in allF.columns if c not in ['household_key','snapshot_day','future_spend_4w']]
tr = df[df.snapshot_day<=375]; te = df[df.snapshot_day.isin([403,431])]
ytr = tr['future_spend_4w'].values; yte = te['future_spend_4w'].values

def train(FEATS, alpha=0.5, lr=0.03, rounds=1200, hl=None, depth=6, mcw=25, ss=0.7, cs=0.7, seed=1, ref=375):
    wt = np.ones(len(tr)) if hl is None else 0.5**((ref-tr['snapshot_day'].values)/hl)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=rounds, learning_rate=lr,
                         max_depth=depth, min_child_weight=mcw, subsample=ss, colsample_bytree=cs, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[FEATS], ytr, sample_weight=wt)
    return m

# base honest preds (2 seeds, no decay)
Ps = []
for s in (1,2):
    m = train(BASE, seed=s); Ps.append(np.clip(m.predict(te[BASE]),0,None))
P = np.mean(Ps,axis=0)
res = yte - P
print("honest bias:", res.mean().round(2), "median resid:", np.median(res).round(2))
print("raw MAE:", np.abs(res).mean().round(3))
for c in [5,10,15,20,np.median(res)]:
    print(f"shift +{c:.1f}: MAE={np.abs(P+c-yte).mean():.3f}")
# alpha variants
for a in [0.52, 0.55, 0.58]:
    Ps=[]
    for s in (1,2):
        m = train(BASE, alpha=a, seed=s); Ps.append(np.clip(m.predict(te[BASE]),0,None))
    Pa = np.mean(Ps,axis=0)
    print(f"alpha={a}: MAE={np.abs(Pa-yte).mean():.3f} bias={(Pa-yte).mean():.1f}")
# top70 feats + no decay (best pruning) with shift
imp = pd.Series(train(BASE).feature_importances_, index=BASE).sort_values(ascending=False)
F70 = list(imp.head(70).index)
Ps=[]
for s in (1,2):
    m = train(F70, seed=s); Ps.append(np.clip(m.predict(te[F70]),0,None))
P70 = np.mean(Ps,axis=0)
print("top70 nodecay raw:", np.abs(P70-yte).mean().round(3), "| +10:", np.abs(P70+10-yte).mean().round(3), "| +15:", np.abs(P70+15-yte).mean().round(3))
