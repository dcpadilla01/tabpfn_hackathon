import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet')
tt = agent_api.train_targets()
df = e7.merge(cand.drop(columns=['snapshot_day']), on='household_key', how='left').merge(tt, on=['household_key','snapshot_day'])
print(df.shape)

def prep(d):
    X = d.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    cats=[c for c in X.columns if X[c].dtype==object or str(X[c].dtype)=='category']
    X = pd.get_dummies(X, columns=cats, dummy_na=True).astype(np.float64)
    return X.fillna(X.median()).fillna(0)

def ridge_fit(Xtr,ytr,alpha):
    mu,sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[0,0]-=alpha
    return np.linalg.solve(A, Z.T@ytr), mu, sd
def ridge_pred(w,mu,sd,X):
    return np.c_[np.ones(len(X)), (X-mu)/sd]@w

y = df.future_spend_4w.values
sd_ = df.snapshot_day.values
tr = sd_<=403; te = sd_==431
cand_cols = ['dec_spend14','dec_spend7','dec_trips','gap_mean','gap_std','gap_med','ntrip112','pl_share','disc_share2','qty112','unit_price','macro_ratio']

def evalset(cols, alpha=10.0):
    X = prep(df[cols+['household_key','snapshot_day','future_spend_4w']])
    w,mu,s = ridge_fit(X[tr], y[tr], alpha)
    return np.abs(ridge_pred(w,mu,s,X[te]) - y[te]).mean()

base = evalset([c for c in df.columns if c not in cand_cols+['future_spend_4w']])
print('base(e7) 431 MAE:', round(base,3))
for c in cand_cols:
    m = evalset([c for c in df.columns if c not in cand_cols+['future_spend_4w']] + [c])
    print(f'+{c:14s} {m:8.3f}  delta {m-base:+.3f}')
allc = evalset([c for c in df.columns if c not in ['future_spend_4w']])
print('all e7+cand:', round(allc,3))
