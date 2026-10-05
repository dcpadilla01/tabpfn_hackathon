
import agent_api, pandas as pd, numpy as np, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats.drop(columns=['index']), on=['household_key','snapshot_day'], how='left')

TRAIN_DAYS = [95,123,151,179,207,235,263,291,319,347,375,403,431]
print("target mean/median by snapshot day:")
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']).round(1))

FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]
print("\nn features:", len(FE))

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

tr = df[df.snapshot_day<=403]
va = df[df.snapshot_day==431]
Xtr,ytr = make_xy(tr); Xva,yva = make_xy(va)
print("train rows:", len(tr), "val rows:", len(va), "val zero frac:", (yva==0).mean())

def fit_q(Xtr,ytr,depth=4,mcw=20,nest=600,lr=0.05):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
                         max_depth=depth, min_child_weight=mcw, n_estimators=nest,
                         learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                         n_jobs=4, random_state=0)
    m.fit(Xtr,ytr)
    return m

m = fit_q(Xtr,ytr)
p = m.predict(Xva)
blend = Xva['exp4w_blend'].values
final = np.clip(0.7*p + 0.3*blend, 0, None)
print("\nlocal val (431): model-only MAE:", np.abs(p-yva).mean())
print("local val blend0.7 MAE:", np.abs(final-yva).mean())
print("persistence-only MAE:", np.abs(blend-yva).mean())
print("mean pred:", final.mean(), "mean y:", yva.mean())
