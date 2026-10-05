
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = e2.merge(t, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values

X = m[[c for c in e2.columns if c not in ('household_key','snapshot_day')]].copy()
cat = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
X = pd.get_dummies(X, columns=cat, dummy_na=True).astype(float)
X = X.replace([np.inf,-np.inf], np.nan)
print("nan cols:", X.isna().sum()[X.isna().sum()>0])
X = X.fillna(X.median(numeric_only=True)).fillna(0)
mu, sd = X.mean(), X.std().replace(0,1)
Xz = ((X-mu)/sd).values
A = Xz[~vd]; ya = np.log1p(y[~vd])
w = np.linalg.solve(A.T@A + 30*np.eye(A.shape[1]), A.T@ya)
p = np.clip(np.expm1(Xz[vd]@w), 0, None)
print("finite preds:", np.isfinite(p).mean(), p.min(), p.max())
print("MAE:", np.abs(p - y[vd]).mean())
