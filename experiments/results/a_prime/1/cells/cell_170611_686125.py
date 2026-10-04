
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)

feats = [c for c in u.columns if c not in ('household_key','snapshot_day')]
X = m[feats].copy()
# numeric only, median impute, standardize
num = X.select_dtypes(include=[np.number]).columns
X = X[num]
X = X.fillna(X.median())
mu, sd = X.mean(), X.std().replace(0,1)
Xs = ((X-mu)/sd).values
Xs = np.hstack([Xs, np.ones((len(Xs),1))])

def ridge_fit(X, y, lam=10.0):
    d = X.shape[1]
    A_ = X.T@X + lam*np.eye(d); A_[-1,-1] -= lam  # don't penalize intercept much
    return np.linalg.solve(A_, X.T@y)

def mae(p,y): return np.mean(np.abs(p-y))

days = sorted(m.snapshot_day.unique())
oof = np.zeros(len(m))
for d in days:
    tr = m.snapshot_day != d; te = ~tr
    w = ridge_fit(Xs[tr.values], y[tr.values], lam=20.0)
    oof[te.values] = Xs[te.values]@w
print("LOSO ridge MAE:", round(mae(oof,y),3))
g = pd.DataFrame({'d':m.snapshot_day,'err':oof-y,'ae':np.abs(oof-y)})
print(g.groupby('d').agg(bias=('err','mean'), mae=('ae','mean')).round(2))
print("mean bias:", g.err.mean().round(2))
