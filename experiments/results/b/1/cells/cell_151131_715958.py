import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e001_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')

def onehot(df, cats):
    parts = []
    for c in cats:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        parts.append(d.values.astype(float))
    return parts

def fit_ridge(Xtr, ytr, lam=1.0):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Z = (Xtr-mu)/sd
    Z = np.hstack([Z, np.ones((len(Z),1))])
    A = Z.T@Z + lam*np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T@ytr), mu, sd

def predict(model, Xte):
    w, mu, sd = model
    Z = (Xte-mu)/sd
    Z = np.hstack([Z, np.ones((len(Z),1))])
    return Z@w

def prep(df, num_cols, cats, target=True):
    Xnum = df[num_cols].astype(float).values.copy()
    Xnum[np.isnan(Xnum)] = 0.0
    parts = [Xnum] + onehot(df, cats)
    X = np.hstack(parts)
    y = df[target_col].values if target else None
    return X, y

target_col='future_spend_4w'
cats = ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc']
num_e000 = ['has_demographics','snapshot_day_index','week_of_year']
num_e001 = [c for c in t.columns if t[c].dtype.kind in 'if' and c not in ('household_key','snapshot_day','index')]

for name, num in [('E000', num_e000), ('E001', num_e001)]:
    X, y = prep(df, num, cats)
    tr = df.snapshot_day <= 403
    va = df.snapshot_day == 431
    m = fit_ridge(X[tr.values], y[tr.values], lam=10.0)
    p = predict(m, X[va.values])
    print(f'{name}: local holdout(431) MAE={np.abs(p-y[va.values]).mean():.3f}  (harness: {"92.531" if name=="E000" else "63.025"})')