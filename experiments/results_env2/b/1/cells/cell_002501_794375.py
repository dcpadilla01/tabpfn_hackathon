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
cn = agent_api.load_saved('cand_new1.parquet')
tt = agent_api.train_targets()
gcols = [c for c in t16.columns if c.startswith('g_')]
base = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').merge(
    t16[['household_key','snapshot_day']+gcols], on=['household_key','snapshot_day'])
base = base.sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = base.snapshot_day.values; y = base.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
Xb,_ = build_matrix(base)
print('base E008+g : %.4f' % ridge(Xb, snap, y, FIT, EVAL))
ncols = [c for c in cn.columns if c not in ('household_key','snapshot_day')]
m2 = base.merge(cn, on=['household_key','snapshot_day'])
X2,_ = build_matrix(m2)
print('base+cand   : %.4f' % ridge(X2, snap, y, FIT, EVAL))
fams = {'A_tail':[c for c in ncols if c.startswith(('n_big','bigshare','top3b','maxline','hi_item'))],
        'B_dabs':[c for c in ncols if c.startswith('dabs84')],
        'D_dorm':['dorm_r','act_exp','zeros_last3'],
        'E_disc':['cd84_r','rd84_r'],
        'F_tod':['eve_share84','morn_share84'],
        'H_up':['up84'],
        'C_rank':['r_sp84','r_dsl','r_tr84']}
for k,v in fams.items():
    Xk,_ = build_matrix(m2.drop(columns=v))
    print('  drop %-7s: %.4f' % (k, ridge(Xk, snap, y, FIT, EVAL)))
# residual correlations
istr = np.isin(snap, FIT)
mu,sd = Xb[istr].mean(0), Xb[istr].std(0)+1e-9
Zb=(Xb-mu)/sd; Zb=np.column_stack([np.ones(len(Zb)),Zb])
A=Zb[istr].T@Zb[istr]+200*np.eye(Zb.shape[1]); A[0,0]-=200
w=np.linalg.solve(A, Zb[istr].T@y[istr])
res = y - Zb@w
Xn,_ = build_matrix(m2[ncols])
print('\n|corr| with residual (full rows):')
for i,c in enumerate(ncols):
    v = Xn[:,i]
    if v.std()<1e-12: print('  %-16s const' % c); continue
    print('  %-16s %.4f' % (c, abs(np.corrcoef(v, res)[0,1])))
