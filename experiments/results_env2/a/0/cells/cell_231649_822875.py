
import pandas as pd, numpy as np, agent_api, xgboost as xgb

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
resid = yv - base
print("resid mean:", round(resid.mean(),2), " MAE:", round(np.abs(resid).mean(),2))

# correlation of signed residual with each feature
cors = []
for c in FE:
    x = Xv[c].values.astype(float)
    ok = np.isfinite(x)
    if ok.sum()>100 and np.std(x[ok])>0:
        r = np.corrcoef(x[ok], resid[ok])[0,1]
        cors.append((c, r))
cors = sorted(cors, key=lambda t: -abs(t[1]))
print("\ntop |corr| with signed residual:")
for c,r in cors[:20]: print(f"  {c:22s} {r:+.3f}")

# reconstruction check: lag0 = spend in (s-28, s]? compare with spend_28
v = agent_api.snapshot(as_of_day=431)
tx = v.table('transactions')[['household_key','day','sales_value']]
s = 431
lag0 = tx[(tx.day>s-28)&(tx.day<=s)].groupby('household_key').sales_value.sum()
chk = feats[(feats.snapshot_day==431)].set_index('household_key')['spend_28']
cmp = pd.concat([lag0.rename('mine'), chk.rename('saved')], axis=1).dropna()
print("\nlag0 recon match:", np.abs(cmp.mine-cmp.saved).max(), " n:", len(cmp))
lag1 = tx[(tx.day>s-56)&(tx.day<=s-28)].groupby('household_key').sales_value.sum()
chk1 = feats[(feats.snapshot_day==431)].set_index('household_key')['spend28_lag1']
cmp1 = pd.concat([lag1.rename('mine'), chk1.rename('saved')], axis=1).dropna()
print("lag1 recon match:", np.abs(cmp1.mine-cmp1.saved).max(), " n:", len(cmp1))
