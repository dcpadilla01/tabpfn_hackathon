import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e7 = agent_api.load_saved('e007_lagseq.parquet')
cand = agent_api.load_saved('cand1.parquet')
tt = agent_api.train_targets()
cand = cand.rename(columns={'qty112':'c_qty112'})
df = e7.merge(cand.drop(columns=['snapshot_day']), on='household_key', how='left').merge(tt, on=['household_key','snapshot_day'])

def prep(d):
    X = d.drop(columns=['household_key','snapshot_day','future_spend_4w'], errors='ignore')
    cats=[c for c in X.columns if X[c].dtype==object or str(X[c].dtype)=='category']
    X = pd.get_dummies(X, columns=cats, dummy_na=True).astype(np.float64)
    return X.fillna(X.median()).fillna(0)
def rfit(Xtr,ytr,alpha):
    mu,sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = np.c_[np.ones(len(Xtr)), (Xtr-mu)/sd]
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); A[0,0]-=alpha
    return np.linalg.solve(A, Z.T@ytr), mu, sd
def rpred(w,mu,sd,X): return np.c_[np.ones(len(X)), (X-mu)/sd]@w

y = df.future_spend_4w.values; sd_ = df.snapshot_day.values
tr = sd_<=403; te = sd_==431
cand_cols = ['dec_spend14','dec_spend7','dec_trips','gap_mean','gap_std','gap_med','ntrip112','pl_share','disc_share2','c_qty112','unit_price','macro_ratio']
def ev(cols, alpha=10.0):
    X = prep(df[cols+['household_key','snapshot_day','future_spend_4w']])
    w,mu,s = rfit(X[tr], y[tr], alpha)
    return np.abs(rpred(w,mu,s,X[te]) - y[te]).mean()
base_cols = [c for c in df.columns if c not in cand_cols+['future_spend_4w']]
base = ev(base_cols); print('base:', round(base,3))
for c in ['c_qty112','unit_price','macro_ratio']:
    print(f'+{c:12s} {ev(base_cols+[c]):8.3f}  delta {ev(base_cols+[c])-base:+.3f}')
combos = {
 'dec14+dec7+dectrips': ['dec_spend14','dec_spend7','dec_trips'],
 'dec14+ntrip112': ['dec_spend14','ntrip112'],
 'allgood': ['dec_spend14','dec_spend7','dec_trips','ntrip112'],
 'allgood+macro': ['dec_spend14','dec_spend7','dec_trips','ntrip112','macro_ratio'],
 'all12': cand_cols,
}
for k,v in combos.items():
    m = ev(base_cols+v)
    print(f'{k:22s} {m:8.3f}  delta {m-base:+.3f}')
