
import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
holdout = [347, 375, 403, 431]
inner = [d for d in sorted(df.snapshot_day.unique()) if d not in holdout]
m_tr = df.snapshot_day.isin(inner).values
m_ho = df.snapshot_day.isin(holdout).values
Xtr = df.loc[m_tr, feat_cols].astype(float).values
Xho = df.loc[m_ho, feat_cols].astype(float).values
ytr = df.loc[m_tr,'future_spend_4w'].values
yho = df.loc[m_ho,'future_spend_4w'].values

base = dict(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8)
def mae(p): return round(np.abs(p-yho).mean(),3)

# reference: median model
t0=time.time()
m = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr,ytr)
p_med = np.clip(m.predict(Xho),0,None)
print('med ref MAE', mae(p_med), round(time.time()-t0,1),'s')

# A) winsorized target (median objective)
for q in [0.95, 0.98, 0.99]:
    cap = np.quantile(ytr, q)
    m = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr, np.minimum(ytr,cap))
    print(f'winsor@{q} (cap={cap:.0f}) MAE', mae(np.clip(m.predict(Xho),0,None)))

# B) two-part model
t0=time.time()
clf = xgb.XGBClassifier(**{**base,'objective':'binary:logistic','eval_metric':'logloss'}).fit(Xtr,(ytr>0).astype(int))
p_pos = clf.predict_proba(Xho)[:,1]
reg = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr[ytr>0], ytr[ytr>0])
p_amt = np.clip(reg.predict(Xho),0,None)
p2 = p_pos*p_amt
print('two-part MAE', mae(p2), 'pos-rate pred', round(p_pos.mean(),3), 'actual', round((yho>0).mean(),3), round(time.time()-t0,1),'s')
# two-part with calibrated exponent on p_pos
for e in [0.7,0.85,1.15]:
    print(f'  two-part p^{e} MAE', mae((p_pos**e)*p_amt))
# blend two-part with median model
for w in [0.2,0.3,0.4,0.5]:
    print(f'  blend med+{w}*twopart MAE', mae((1-w)*p_med + w*p2))

# C) global shrink of median model
for s in [0.9,0.95,1.05,1.1]:
    print(f'shrink {s} MAE', mae(p_med*s))
