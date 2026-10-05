import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
t8  = agent_api.load_saved('e008_level_shape.parquet')
t6  = agent_api.load_saved('e006_cadence.parquet')
t7  = agent_api.load_saved('e007_temporal.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt  = agent_api.train_targets()
key = ['household_key','snapshot_day']
def uniq(t, base_cols):
    return [c for c in t.columns if c not in base_cols]
cols8  = uniq(t8,  key)
cols6  = [c for c in uniq(t6, key)  if c not in cols8]
cols7  = [c for c in uniq(t7, key)  if c not in cols8+cols6]
cols16 = [c for c in uniq(t16, key) if c not in cols8+cols6+cols7]
print('counts:', len(cols8), len(cols6), len(cols7), len(cols16))
m = t8[key+cols8].merge(t6[key+cols6], on=key).merge(t7[key+cols7], on=key).merge(t16[key+cols16], on=key)
print('grand table:', m.shape)
agent_api.save_table(m, 'e017_grand.parquet')

# screen on late-train holdout
def build_matrix(df, drop=('household_key','snapshot_day','future_spend_4w')):
    drop = list(drop)
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool') and c not in drop]
    num_cols = [c for c in df.columns if c not in cat_cols + drop]
    Xs = []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df)))
    for c in cat_cols:
        Xs.append(pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True).values.astype(float))
    return np.column_stack(Xs)
def ridge(X, snap, y, fit_snaps, eval_snaps, lam=200.0):
    istr = np.isin(snap, fit_snaps); isv = np.isin(snap, eval_snaps)
    mu, sd = X[istr].mean(0), X[istr].std(0)+1e-9
    Z = (X-mu)/sd; Z = np.column_stack([np.ones(len(Z)), Z])
    A = Z[istr].T@Z[istr] + lam*np.eye(Z.shape[1]); A[0,0]-=lam
    w = np.linalg.solve(A, Z[istr].T@y[istr])
    return np.abs(np.clip(Z[isv]@w,0,None)-y[isv]).mean()
b = t8.merge(tt, on=key, how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = b.snapshot_day.values; y = b.future_spend_4w.values.astype(float)
FIT=[95,123,151,179,207,235,263,291,319,347,375]; EVAL=[403,431]
g = m.merge(tt, on=key, how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
assert (g.household_key.values==b.household_key.values).all()
print('screen E008      : %.4f' % ridge(build_matrix(b.drop(columns=['future_spend_4w'])), snap, y, FIT, EVAL))
print('screen grand union: %.4f' % ridge(build_matrix(g.drop(columns=['future_spend_4w'])), snap, y, FIT, EVAL))
