
import pandas as pd, numpy as np, time

t15 = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()

drop = {'household_key','snapshot_day','stack_ridge_log','stack_ridge','index'}
feat_cols = [c for c in t15.columns if c not in drop]
cat_cols = ['classification_1','classification_2','classification_3','classification_4',
            'classification_5','homeowner_desc','kid_category_desc']
num_cols = [c for c in feat_cols if c not in cat_cols]

df = t15.merge(tt, on=['household_key','snapshot_day'], how='left')
train_mask = df['future_spend_4w'].notna().values

Xn = df[num_cols].astype(float)
Xn = Xn.fillna(Xn.median())
Xc = pd.get_dummies(df[cat_cols].astype('category'), dummy_na=True, dtype=float)
X = np.hstack([Xn.values, Xc.values])
y = df['future_spend_4w'].values
ly = np.log1p(y)
sd = df['snapshot_day'].values
train_days = sorted(pd.unique(sd[train_mask]))

def ridge_fit_pred(Xtr, ytr, Xte, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    A = (Xtr-mu)/sg; B = (Xte-mu)/sg
    A = np.hstack([A, np.ones((len(A),1))]); B = np.hstack([B, np.ones((len(B),1))])
    d = A.shape[1]
    P = np.eye(d); P[-1,-1] = 0.0
    w = np.linalg.solve(A.T@A + alpha*P, A.T@ytr)
    return B@w

t0=time.time()
oof = np.zeros(train_mask.sum())
for s in train_days:
    m = (sd==s) & train_mask
    mfit = train_mask & (sd!=s)
    oof[m] = ridge_fit_pred(X[mfit], ly[mfit], X[m], 300.0)
mae_log = np.mean(np.abs(np.expm1(oof) - y[train_mask]))
print("LOSO ridge log alpha=300  LOSO-MAE(train):", round(mae_log,3), " time:", round(time.time()-t0,1))
