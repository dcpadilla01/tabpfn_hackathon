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
ALLSNAPS = [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]

# expanding per-household past-target mean (strictly earlier snapshots)
tt2 = tt.sort_values(['household_key','snapshot_day'])
g = tt2.groupby('household_key')
past_sum = g.future_spend_4w.cumsum() - tt2.future_spend_4w
past_cnt = g.cumcount()
te_mean = (past_sum/past_cnt.replace(0,np.nan))
# EWMA over past targets per household (halflife 2 snapshots)
def ewma(sub):
    v = sub.future_spend_4w.values.astype(float)
    out = np.empty(len(v)); acc=np.nan; wsum=0.0
    for i in range(len(v)):
        if i>0:
            wsum = wsum*0.5 + 1.0; acc = (np.nan_to_num(acc,nan=0.0)*0.5 + v[i-1])
            # proper: weights 0.5^k
        out[i] = np.nan if i==0 else np.average(v[:i], weights=0.5**np.arange(i-1,-1,-1))
    return pd.Series(out, index=sub.index)
te_ew = tt2.groupby('household_key', group_keys=False).apply(ewma)
te = pd.DataFrame({'household_key':tt2.household_key,'snapshot_day':tt2.snapshot_day,
                   'te_mean':te_mean.values,'te_cnt':past_cnt.values,'te_ew':te_ew.values})
gm = tt.future_spend_4w.mean()
te['te_mean'] = te.te_mean.fillna(gm); te['te_ew'] = te.te_ew.fillna(gm)

m = base.merge(te, on=['household_key','snapshot_day'], how='left')
m['snap_day'] = m.snapshot_day/100.0
Xb,_ = build_matrix(base)
print('base            : %.4f' % ridge(Xb, snap, y, FIT, EVAL))
X1,_ = build_matrix(m.drop(columns=['te_mean','te_cnt','te_ew']))
print('+snap_day trend : %.4f' % ridge(X1, snap, y, FIT, EVAL))
X2,_ = build_matrix(m.drop(columns=['snap_day']))
print('+te (mean/cnt/ew): %.4f' % ridge(X2, snap, y, FIT, EVAL))
X3,_ = build_matrix(m)
print('+both           : %.4f' % ridge(X3, snap, y, FIT, EVAL))
for lam in [100,400]:
    print('  +both lam %d: %.4f' % (lam, ridge(X3, snap, y, FIT, EVAL, lam)))
# corr of te_mean with y on eval snaps
isv = np.isin(snap, EVAL)
print('te_mean corr w/ y (eval rows): %.4f' % np.corrcoef(m.te_mean[isv], y[isv])[0,1])
print('sp84 corr w/ y (eval rows)   : %.4f' % np.corrcoef(pd.to_numeric(base.sp84,errors='coerce').fillna(0)[isv], y[isv])[0,1])
