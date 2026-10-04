
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

u = A.load_saved('e018_union_full.parquet')
tt = A.train_targets()
m = tt.merge(u, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)

feats = [c for c in u.columns if c not in ('household_key','snapshot_day')]
X = m[feats].select_dtypes(include=[np.number]).fillna(m[feats].select_dtypes(include=[np.number]).median())
mu, sd = X.mean(), X.std().replace(0,1)
Xs = np.hstack([((X-mu)/sd).values, np.ones((len(X),1))])
def ridge_fit(X, y, lam=20.0):
    d = X.shape[1]; A_ = X.T@X + lam*np.eye(d); A_[-1,-1] -= lam
    return np.linalg.solve(A_, X.T@y)
def mae(p,y): return np.mean(np.abs(p-y))

# A) LOSO with spend_l13-family dropped
drop13 = [c for c in feats if c in ('spend_l13','pct_l13','nspend_ly','has_real_l13','l13_over_recent')]
fA = [c for c in feats if c not in drop13]
XA = m[fA].select_dtypes(include=[np.number]).fillna(m[fA].select_dtypes(include=[np.number]).median())
XsA = np.hstack([((XA-XA.mean())/XA.std().replace(0,1)).values, np.ones((len(XA),1))])
oofA = np.zeros(len(m))
for d in sorted(m.snapshot_day.unique()):
    tr = m.snapshot_day != d
    w = ridge_fit(XsA[tr.values], y[tr.values])
    oofA[~tr.values] = XsA[~tr.values]@w
print("LOSO ridge WITHOUT l13-family:", round(mae(oofA,y),3))
print(pd.DataFrame({'d':m.snapshot_day,'err':oofA-y}).groupby('d').agg(bias=('err','mean'),mae=('err',lambda e: np.abs(e).mean())).round(2))

# B) LOSO full (repeat for direct comparison, lam 20)
oofB = np.zeros(len(m))
for d in sorted(m.snapshot_day.unique()):
    tr = m.snapshot_day != d
    w = ridge_fit(Xs[tr.values], y[tr.values])
    oofB[~tr.values] = Xs[~tr.values]@w
print("\nLOSO ridge full:", round(mae(oofB,y),3))
print(pd.DataFrame({'d':m.snapshot_day,'err':oofB-y}).groupby('d').agg(bias=('err','mean'),mae=('err',lambda e: np.abs(e).mean())).round(2))
