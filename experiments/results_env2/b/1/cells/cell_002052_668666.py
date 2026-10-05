import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
def build_matrix(df):
    df = df.copy()
    drop = ['household_key','snapshot_day','future_spend_4w']
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool') and c not in drop]
    num_cols = [c for c in df.columns if c not in cat_cols + drop]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df))); names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names

def ridge(X, snap, y, fit_snaps, eval_snaps, lam=200.0, return_pred=False):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    pv = np.clip(Z[isv]@w, 0, None)
    return (np.abs(pv-y[isv]).mean(), pv) if return_pred else np.abs(pv-y[isv]).mean()

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
gcols = [c for c in t16.columns if c.startswith('g_')]
m = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(
    t16[['household_key','snapshot_day']+gcols], on=['household_key','snapshot_day'])
m = m.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = m.snapshot_day.values; y = m.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]

X8,_  = build_matrix(m.drop(columns=gcols))
Xa,_  = build_matrix(m)
Xg,_  = build_matrix(m[['household_key','snapshot_day']+gcols])
print('E008        : %.4f' % ridge(X8, snap, y, FIT, EVAL))
print('E008+g_grid : %.4f' % ridge(Xa, snap, y, FIT, EVAL))
print('g_grid only : %.4f' % ridge(Xg, snap, y, FIT, EVAL))
for lam in [50,100,200,400]:
    print('  E008+g lam %3d: %.4f' % (lam, ridge(Xa, snap, y, FIT, EVAL, lam)))
# per-snap for E008+g
mae, pv = ridge(Xa, snap, y, FIT, EVAL, 100, True)
isv = np.isin(snap, EVAL)
for s in EVAL:
    k = snap[isv]==s
    print('  snap %d MAE %.2f  predmean %.1f ymean %.1f' % (s, np.abs(pv[k]-y[isv][k]).mean(), pv[k].mean(), y[isv][k].mean()))
