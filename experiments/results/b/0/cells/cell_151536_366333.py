import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet').rename(columns={'qty112':'c_qty112'})
tt = agent_api.train_targets()
df = e7.merge(cand.drop(columns=['snapshot_day']), on='household_key', how='left').merge(tt, on=['household_key','snapshot_day'])
def prep(d):
    X = d.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    cats=[c for c in X.columns if X[c].dtype==object or str(X[c].dtype)=='category']
    return pd.get_dummies(X, columns=cats, dummy_na=True).astype(np.float64).fillna(X.median()).fillna(0) if False else pd.get_dummies(X, columns=cats, dummy_na=True).astype(np.float64).replace([np.inf,-np.inf],np.nan).fillna(0)
def rfit(Xtr,ytr,alpha):
    mu,sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[0,0]-=alpha
    return np.linalg.solve(A, Z.T@ytr), mu, sd
def rpred(w,mu,sd,X): return np.c_[np.ones(len(X)), (X-mu)/sd]@w
y = df.future_spend_4w.values; sd_ = df.snapshot_day.values
cand_cols = ['dec_spend14','dec_spend7','dec_trips','gap_mean','gap_std','gap_med','ntrip112','pl_share','disc_share2','c_qty112','unit_price','macro_ratio']
base_cols = [c for c in df.columns if c not in cand_cols+['future_spend_4w']]
def ev(cols, fitmax, evals, alpha=10.0):
    X = prep(df[cols+['household_key','snapshot_day','future_spend_4w']])
    w,mu,s = rfit(X[sd_<=fitmax], y[sd_<=fitmax], alpha)
    m = sd_.isin(evals)
    return np.abs(rpred(w,mu,s,X[m]) - y[m]).mean()
for fitmax, evals in [(375,[403,431]), (403,[431])]:
    b = ev(base_cols, fitmax, evals)
    a = ev(base_cols+['dec_spend14','dec_spend7','dec_trips','ntrip112'], fitmax, evals)
    c = ev(base_cols+cand_cols, fitmax, evals)
    print(f'fit<={fitmax} eval{evals}: base {b:.3f} | +dec4 {a:.3f} ({a-b:+.3f}) | +all12 {c:.3f} ({c-b:+.3f})')
