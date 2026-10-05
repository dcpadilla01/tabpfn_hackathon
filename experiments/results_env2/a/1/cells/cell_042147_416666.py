import agent_api as api, pandas as pd, numpy as np, time
import xgboost as xgb
allF = api.load_saved('allF.parquet')
tt = api.train_targets()[['household_key','snapshot_day','future_spend_4w']]
df = allF.drop(columns=['future_spend_4w']).merge(tt, on=['household_key','snapshot_day'])
BASE = [c for c in allF.columns if c not in ['household_key','snapshot_day','future_spend_4w']]
tr = df[df.snapshot_day<=375]; te = df[df.snapshot_day.isin([403,431])]
ytr = tr['future_spend_4w'].values; yte = te['future_spend_4w'].values

def train(FEATS, lr=0.03, rounds=1200, hl=None, depth=6, mcw=25, ss=0.7, cs=0.7, seed=1, ref=375):
    wt = np.ones(len(tr)) if hl is None else 0.5**((ref-tr['snapshot_day'].values)/hl)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=rounds, learning_rate=lr,
                         max_depth=depth, min_child_weight=mcw, subsample=ss, colsample_bytree=cs, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[FEATS], ytr, sample_weight=wt)
    return m

# 1) bias by prediction bucket (base config, seed 1)
m = train(BASE)
P = np.clip(m.predict(te[BASE]),0,None)
b = pd.qcut(P, 8, duplicates='drop')
g = pd.DataFrame({'p':P,'y':yte,'b':b}).groupby('b',observed=True).agg(n=('y','size'),pred=('p','mean'),act=('y','mean'),mae=('p',lambda x: 0))
g['mae'] = pd.DataFrame({'p':P,'y':yte,'b':b}).groupby('b',observed=True).apply(lambda x: np.abs(x['p']-x['y']).mean(), include_groups=False)
print("Bias by prediction octile (honest 403/431):"); print(g.round(1))

# 2) feature importance screening: train once, keep top-k
imp = pd.Series(m.feature_importances_, index=BASE).sort_values(ascending=False)
print("\ntop15:", list(imp.head(15).round(3).items()))
for k in [40, 70]:
    Fk = list(imp.head(k).index)
    preds=[]
    for s in (1,2):
        mm = train(Fk, seed=s)
        preds.append(np.clip(mm.predict(te[Fk]),0,None))
    Pk = np.mean(preds,axis=0)
    print(f"top{k} feats: honest MAE={np.abs(Pk-yte).mean():.3f}")
# 3) combined small winners: no decay + snapshot_day + hl277 ensemble
F3 = BASE + ['snapshot_day']
preds=[]
for cfg in [dict(hl=None), dict(hl=277), dict(hl=None), dict(hl=277)]:
    for s in (1,2):
        mm = train(F3, seed=s, **cfg)
        preds.append(np.clip(mm.predict(te[F3]),0,None))
P3 = np.mean(preds,axis=0)
print(f"diverse(no-decay+hl277, +day): honest MAE={np.abs(P3-yte).mean():.3f}")
