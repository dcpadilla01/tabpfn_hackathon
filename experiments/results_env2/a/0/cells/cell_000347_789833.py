import pandas as pd, numpy as np, agent_api, xgboost as xgb
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
print('dtypes non-numeric:', [c for c in feats.columns if feats[c].dtype.kind not in 'fi'])
# encode any object cols
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feats2 = df.drop(columns=['future_spend_4w'])
y = df['future_spend_4w'].values
feat_cols = [c for c in feats2.columns if c not in ('household_key','snapshot_day')]
X = feats2[feat_cols].astype(float).values

def fit_q(Xtr, ytr, Xte, alpha=0.5, depth=5, mcw=40, lr=0.08, n=400):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
                         max_depth=depth, min_child_weight=mcw, learning_rate=lr,
                         n_estimators=n, subsample=0.9, colsample_bytree=0.8,
                         n_jobs=8, tree_method='hist')
    m.fit(Xtr, ytr); return m.predict(Xte)

def fit_c(Xtr, ytr, Xte):
    m = xgb.XGBClassifier(objective='binary:logistic', max_depth=4, min_child_weight=40,
                          learning_rate=0.08, n_estimators=300, subsample=0.9,
                          colsample_bytree=0.8, n_jobs=8, tree_method='hist')
    m.fit(Xtr, ytr); return m.predict_proba(Xte)[:,1]

# internal validation: train on snapshots <=403, test on 431
tr = df.snapshot_day <= 403
te = df.snapshot_day == 431
itr, ite = np.where(tr)[0], np.where(te)[0]
Xtr, ytr, Xte, yte = X[itr], y[itr], X[ite], y[ite]
print('train rows', len(itr), 'test rows', len(ite), 'test MAE of exp4w_blend:', round(np.abs(feats2.exp4w_blend.values[ite]-yte).mean(),3))

# A: plain quantile model (E011 style)
pa = fit_q(Xtr, ytr, Xte)
print('A quantile all:', round(np.abs(pa-yte).mean(),3))
# B: hurdle p*q
pc = fit_c(Xtr, (ytr>0).astype(int), Xte)
pos = ytr>0
pq = fit_q(Xtr[pos], ytr[pos], Xte)
pb = pc*pq
print('B hurdle p*q:', round(np.abs(pb-yte).mean(),3))
# C: blend model with exp4w_blend
eb = feats2.exp4w_blend.values[ite]
for w in (0.2,0.3,0.4):
    print(f'C blend w={w}:', round(np.abs((1-w)*pa+w*eb-yte).mean(),3))
# D: hurdle + blend
for w in (0.2,0.3):
    print(f'D hurdle+blend w={w}:', round(np.abs((1-w)*pb+w*eb-yte).mean(),3))
# E: p-shifted quantile (0.55)
pq55 = fit_q(Xtr, ytr, Xte, alpha=0.55)
print('E q0.55 all:', round(np.abs(pq55-yte).mean(),3))
