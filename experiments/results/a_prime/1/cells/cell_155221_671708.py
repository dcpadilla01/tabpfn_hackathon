import agent_api as A
import pandas as pd, numpy as np

micro = A.load_saved('micro.parquet')
tt = A.train_targets()
df = micro.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

cols = [c for c in micro.columns if c not in ('household_key','snapshot_day')]
# univariate: corr with y, MAE of column alone (as predictor), corr with |resid| of a simple base
base_feats = ['mspend_l1','mspend_l2','mspend_l3','mspend_l4','mspend_l5','mspend_l6','mtrips_l1','mtrips_l2','mtrips_l3',
              'spend_7d','spend_14d','dsl','n_zero_w6','n_zero_w12','tenure' if 'tenure' in tr else 'mspend_l6']
def fit_w(X, yy, lam=10.0):
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    return np.linalg.solve(Am, Xs.T@yy), mu, sd
def apply_w(w, mu, sd, X):
    return np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])@w

bf = [c for c in ['mspend_l1','mspend_l2','mspend_l3','mspend_l4','mspend_l5','mspend_l6','mtrips_l1','mtrips_l2','mtrips_l3'] if c in tr]
Xb = tr[bf].fillna(0).values.astype(float)
w, mu, sd = fit_w(Xb, y)
res = y - apply_w(w, mu, sd, Xb)
print('base train MAE:', round(float(np.abs(res).mean()),2))

rows=[]
for c in cols:
    v = tr[c].fillna(0).values if tr[c].dtype.kind in 'fc' else tr[c].fillna('NA')
    if tr[c].dtype.kind in 'fc':
        corr = float(np.corrcoef(v, y)[0,1]) if np.std(v)>0 else 0.0
        corr_res = float(np.corrcoef(v, res)[0,1]) if np.std(v)>0 else 0.0
        corr_abs = float(np.corrcoef(v, np.abs(res))[0,1]) if np.std(v)>0 else 0.0
        # alone-MAE
        w1, mu1, sd1 = fit_w(v.reshape(-1,1), y)
        p1 = apply_w(w1, mu1, sd1, v.reshape(-1,1))
        m1 = float(np.abs(p1-y).mean())
        rows.append((c, round(corr,3), round(corr_res,3), round(corr_abs,3), round(m1,2)))
r = pd.DataFrame(rows, columns=['feat','corr_y','corr_res','corr_absres','alone_mae']).sort_values('alone_mae')
print(r.to_string(index=False))
