
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = e2.merge(t, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w'].values
vd = (m['snapshot_day']==431).values
print("pseudo-val rows:", vd.sum())

def prep(df):
    X = df.copy()
    cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    X = pd.get_dummies(X, columns=cat, dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return ((X - X.mean()) / X.std().replace(0,1)).values

def ridge_eval(Xdf, lam=30.0):
    Xz = prep(Xdf)
    A = Xz[~vd]; ya = np.log1p(y[~vd])
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ya)
    p = np.clip(np.expm1(Xz[vd]@w), 0, None)
    return np.abs(p - y[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print("E002 proxy MAE:", ridge_eval(m[base_cols]))
