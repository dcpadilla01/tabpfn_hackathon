import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
def build_matrix(df, drop=('household_key','snapshot_day','future_spend_4w')):
    drop = list(drop)
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
def ridge(X, snap, y, fit_snaps, eval_snaps, lam=200.0):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    return np.abs(np.clip(Z[isv]@w,0,None)-y[isv]).mean()

t8 = agent_api.load_saved('e008_level_shape.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt = agent_api.train_targets()
gcols = [c for c in t16.columns if c.startswith('g_')]
base = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(
    t16[['household_key','snapshot_day']+gcols], on=['household_key','snapshot_day'])
base = base.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = base.snapshot_day.values; y = base.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
Xb,_ = build_matrix(base)
print('base: %.4f' % ridge(Xb, snap, y, FIT, EVAL))

def numcol(df, c):
    v = pd.to_numeric(df[c], errors='coerce')
    return v.fillna(v.median()).values.astype(float)

extras = []
# 1) binned dummies of log1p(sp84): 12 fixed-width bins on log scale 0..8
lsp = np.log1p(np.clip(numcol(base,'sp84'),0,None))
bins = np.clip(((lsp/0.7).astype(int)), 0, 11)
for b in range(12):
    extras.append(('bin84_%d'%b, (bins==b).astype(float)))
# same for sp28
lsp28 = np.log1p(np.clip(numcol(base,'sp28'),0,None))
b28 = np.clip((lsp28/0.7).astype(int),0,11)
for b in range(12):
    extras.append(('bin28_%d'%b, (b28==b).astype(float)))
# 2) interactions with snap_day
sdn = snap/400.0
for c in ['sp84','sp28','sp364','z_med4w_hist','g_wmean','newma56','days_since_last']:
    v = numcol(base,c); v = (v-v.mean())/(v.std()+1e-9)
    extras.append(('ia_%s'%c, v*sdn))
# 3) pairwise interactions of top-3
v1 = numcol(base,'sp84'); v1=(v1-v1.mean())/(v1.std()+1e-9)
v2 = numcol(base,'days_since_last'); v2=(v2-v2.mean())/(v2.std()+1e-9)
v3 = numcol(base,'g_wmean'); v3=(v3-v3.mean())/(v3.std()+1e-9)
extras.append(('ia_l_r', v1*v2)); extras.append(('ia_l_w', v1*v3)); extras.append(('ia_r_w', v2*v3))
E = np.column_stack([e[1] for e in extras]); enames=[e[0] for e in extras]
X1 = np.column_stack([Xb, E])
print('base+bins+interactions: %.4f' % ridge(X1, snap, y, FIT, EVAL))
# bins only
X2 = np.column_stack([Xb, E[:, :24]])
print('base+bins only        : %.4f' % ridge(X2, snap, y, FIT, EVAL))
# interactions only
X3 = np.column_stack([Xb, E[:, 24:]])
print('base+interactions only: %.4f' % ridge(X3, snap, y, FIT, EVAL))
for lam in [400, 800]:
    print('  all lam %d: %.4f' % (lam, ridge(X1, snap, y, FIT, EVAL, lam)))
