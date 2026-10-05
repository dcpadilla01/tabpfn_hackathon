import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def build_matrix(df):
    df = df.copy()
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool')]
    num_cols = [c for c in df.columns if c not in cat_cols + ['household_key','snapshot_day']]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df)))
        names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names

def ridge_eval(X, snap, y, lam=200.0, verbose=True):
    istr = np.isin(snap, [95,123,151,179,207,235,263,291,319,347,375,403,431])
    isv  = np.isin(snap, [459,487,515,543])
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd
    Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    pv = np.clip(Z[isv]@w, 0, None)
    mae = np.abs(pv-y[isv]).mean()
    if verbose: print('val MAE %.4f  nfeat %d' % (mae, X.shape[1]))
    return mae

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
m = tt.merge(t8, on=['household_key','snapshot_day']).merge(
    t16[[c for c in t16.columns if c.startswith('g_')]+['household_key','snapshot_day']], on=['household_key','snapshot_day'])
m = m.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = m.snapshot_day.values; y = m.future_spend_4w.values.astype(float)
X, names = build_matrix(m)
print('E008+grid screener:'); mae_all = ridge_eval(X, snap, y)
# per-snapshot MAE
istr = np.isin(snap,[95,123,151,179,207,235,263,291,319,347,375,403,431]); isv=np.isin(snap,[459,487,515,543])
mu,sd = X[istr].mean(0), X[istr].std(0)+1e-9; Z=(X-mu)/sd; Z=np.column_stack([np.ones(len(Z)),Z])
A=Z[istr].T@Z[istr]+200*np.eye(Z.shape[1]); A[0,0]-=200; w=np.linalg.solve(A,Z[istr].T@y[istr])
pv=np.clip(Z[isv]@w,0,None)
sv=snap[isv]
for s in [459,487,515,543]:
    k=sv==s; print('  snap %d MAE %.2f (n=%d)' % (s, np.abs(pv[k]-y[isv][k]).mean(), k.sum()))
