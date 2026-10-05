import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
DROP = ['household_key','snapshot_day','future_spend_4w']
BASE = [c for c in allF.columns if c not in DROP]
tr = allF[allF.snapshot_day<=375]
va = allF[allF.snapshot_day>=459]
ytr = tr['future_spend_4w'].values
print("train rows:", len(tr), "val rows:", len(va))

def fit(F, alpha, seed, rounds=1200, lr=0.03, depth=6, mcw=25, ss=0.7, cs=0.7):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=rounds, learning_rate=lr,
                         max_depth=depth, min_child_weight=mcw, subsample=ss, colsample_bytree=cs,
                         n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[F], ytr)  # uniform weights: no time decay
    return m

# feature pruning from a seed-1 alpha=.5 model
m0 = fit(BASE, 0.5, 1)
imp = pd.Series(m0.feature_importances_, index=BASE).sort_values(ascending=False)
F70 = list(imp.head(70).index)

t0=time.time()
preds=[]
for s in range(1,9):
    m = fit(F70, 0.52, s)
    preds.append(np.clip(m.predict(va[F70]),0,None))
P = np.mean(preds,axis=0)
print(f"8 seeds done ({time.time()-t0:.0f}s)  val pred mean={P.mean():.1f} median={np.median(P):.1f} min={P.min():.2f} max={P.max():.1f} finite={np.isfinite(P).all()}")

out = va[['household_key','snapshot_day']].copy()
out['prediction'] = P
path = api.save_table(out, 'e018_preds.parquet')
print("saved:", path, out.shape)
