
import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e012_basket_shape.parquet')
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
ycol = agent_api.TARGET

def prep_df(t):
    df = t.set_index(['household_key','snapshot_day']).join(tt[ycol])
    num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].astype(float); X = X.fillna(X.median())
    y = df[ycol].astype(float)
    return X, y

X, y = prep_df(base)
seeds = (375,403,431)
tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seeds]
itr = X.index.get_level_values(1).isin(tr_days).values
iva = X.index.get_level_values(1).isin(seeds).values
print('n train rows', itr.sum(), 'n seed rows', iva.sum())
mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
Xs = ((X-mu)/sd).values
Xtr = np.c_[np.ones(itr.sum()), Xs[itr]]; Xva = np.c_[np.ones(iva.sum()), Xs[iva]]
ytr = np.log1p(y[itr]).values; yva = y[iva].values
for a in (1,3,10,30,100,300,1000):
    A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
    w = np.linalg.solve(A, Xtr.T@ytr)
    p = np.clip(np.expm1(Xva@w), 0, None)
    print(f'alpha={a:5d}  logMAE={np.abs(p-yva).mean():.3f}')
# also raw-target variant
ytr2 = y[itr].values
for a in (100,300,1000):
    A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
    w = np.linalg.solve(A, Xtr.T@ytr2)
    p = np.clip(Xva@w, 0, None)
    print(f'alpha={a:5d}  rawMAE={np.abs(p-yva).mean():.3f}')
