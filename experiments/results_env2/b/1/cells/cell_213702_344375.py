
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

e2 = agent_api.load_saved('e002_mix.parquet')
t = agent_api.train_targets()
m = t.merge(e2, on=['household_key','snapshot_day'], how='left').dropna(subset=['future_spend_4w'])
y = m['future_spend_4w'].values
vd = m['snapshot_day'].isin(agent_api.snapshot_days()['validation']).values

def ridge_eval(df, target, seed=0):
    X = df.copy()
    cat = [c for c in X.columns if X[c].dtype == object]
    X = pd.get_dummies(X, columns=cat, dummy_na=True)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    mu, sd = X.mean(), X.std().replace(0,1)
    X = ((X-mu)/sd).values
    ly = np.log1p(target)
    lam = 30.0
    A = X[vd==False]; ya = ly[vd==False]
    Xt = X[vd]; 
    I = np.eye(A.shape[1])
    w = np.linalg.solve(A.T@A + lam*I, A.T@ya)
    p = np.expm1(Xt@w)
    return np.abs(p - target[vd]).mean()

base_cols = [c for c in e2.columns if c not in ('household_key','snapshot_day')]
print("E002 proxy MAE:", ridge_eval(m[base_cols], y))
