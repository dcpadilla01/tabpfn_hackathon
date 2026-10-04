import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

e7 = agent_api.load_saved('e007_lagseq.parquet')
tt = agent_api.train_targets()
df = e7.merge(tt, on=['household_key','snapshot_day'], how='inner')
print(df.shape)

ycol='future_spend_4w'
def prep(df):
    X = df.drop(columns=['household_key','snapshot_day',ycol], errors='ignore')
    cats=[]
    for c in X.columns:
        if X[c].dtype==object or str(X[c].dtype)=='category':
            cats.append(c)
    X = pd.get_dummies(X, columns=cats, dummy_na=True)
    X = X.astype(np.float64)
    # fillna with train medians
    med = X.median()
    X = X.fillna(med).fillna(0)
    return X

def ridge_fit(Xtr, ytr, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = (Xtr-mu)/sd
    Z = np.c_[np.ones(len(Z)), Z]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[0,0]-=alpha
    w = np.linalg.solve(A, Z.T@ytr)
    return w, mu, sd

def ridge_pred(w, mu, sd, X):
    Z = (X-mu)/sd
    Z = np.c_[np.ones(len(Z)), Z]
    return Z@w

X = prep(df)
y = df[ycol].values
sd_ = df.snapshot_day.values
print('X shape', X.shape)

for alpha in [1.0, 10.0, 100.0]:
    # fit on train snaps <=403, eval on 431
    m = sd_<=403
    w,mu,s = ridge_fit(X[m], y[m], alpha)
    p = ridge_pred(w,mu,s,X[sd_==431])
    mae = np.abs(p - y[sd_==431]).mean()
    # log target variant
    w2,mu2,s2 = ridge_fit(X[m], np.log1p(y[m]), alpha)
    p2 = np.expm1(ridge_pred(w2,mu2,s2,X[sd_==431]))
    mae2 = np.abs(p2 - y[sd_==431]).mean()
    print(f'alpha={alpha}: 431 MAE raw={mae:.3f}  logtgt={mae2:.3f}')
