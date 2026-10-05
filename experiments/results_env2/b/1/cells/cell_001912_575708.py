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

def ridge_fit_eval(X, snap, y, fit_snaps, eval_snaps, lam=200.0):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    pv = np.clip(Z[isv]@w, 0, None)
    return np.abs(pv-y[isv]).mean()

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
gcols = ['household_key','snapshot_day']+[c for c in t16.columns if c.startswith('g_')]
m = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(t16[gcols], on=['household_key','snapshot_day'])
m = m.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = m.snapshot_day.values; y = m.future_spend_4w.values.astype(float)
FIT = [95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
print('rows', len(m))

X8, _ = build_matrix(m.drop(columns=[c for c in m.columns if c.startswith('g_')]))
print('E008 only      : %.4f' % ridge_fit_eval(X8, snap, y, FIT, EVAL))
Xall, _ = build_matrix(m)
print('E008 + g_ grid : %.4f' % ridge_fit_eval(Xall, snap, y, FIT, EVAL))
# g_ alone
Xg, _ = build_matrix(m[['household_key','snapshot_day']+[c for c in m.columns if c.startswith('g_')]])
print('g_ grid only   : %.4f' % ridge_fit_eval(Xg, snap, y, FIT, EVAL))
# lam sensitivity on E008
for lam in [50,100,200,400,800]:
    print('  E008 lam %4d: %.4f' % (lam, ridge_fit_eval(X8, snap, y, FIT, EVAL, lam)))
