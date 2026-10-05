import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')
def build_matrix(df):
    df = df.copy()
    cat_cols = [c for c in df.columns if str(df[c].dtype) in ('category','object','bool')]
    num_cols = [c for c in df.columns if c not in cat_cols + ['household_key','snapshot_day']]
    Xs, names = [], []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        Xs.append(v.fillna(v.median()).values if v.notna().any() else np.zeros(len(df))); names.append(c)
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype('category'), prefix=c, dummy_na=True)
        Xs.append(d.values.astype(float)); names += list(d.columns)
    return (np.column_stack(Xs) if Xs else np.zeros((len(df),0))), names

t8 = agent_api.load_saved('e008_level_shape.parquet')
tt = agent_api.train_targets()
m = t8.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
snap = m.snapshot_day.values; y = m.future_spend_4w.values.astype(float)
print('y stats by snap:')
print(m.groupby('snapshot_day').future_spend_4w.agg(['mean','median','std','count']).round(1))
# check duplicates in merge
print('dupes:', m.duplicated(['household_key','snapshot_day']).sum())
X8,_ = build_matrix(m)
istr = np.isin(snap,[95,123,151,179,207,235,263,291,319,347,375]); isv=np.isin(snap,[403,431])
mu,sd = X8[istr].mean(0), X8[istr].std(0)+1e-9; Z=(X8-mu)/sd; Z=np.column_stack([np.ones(len(Z)),Z])
A=Z[istr].T@Z[istr]+50*np.eye(Z.shape[1]); A[0,0]-=50; w=np.linalg.solve(A,Z[istr].T@y[istr])
pv=np.clip(Z[isv]@w,0,None)
print('eval y: mean %.2f med %.2f' % (y[isv].mean(), np.median(y[isv])))
print('pred  : mean %.2f med %.2f min %.2f max %.2f' % (pv.mean(), np.median(pv), pv.min(), pv.max()))
print('MAE %.4f  corr %.4f' % (np.abs(pv-y[isv]).mean(), np.corrcoef(pv, y[isv])[0,1]))
# per snapshot
for s in [403,431]:
    k = snap[isv]==s
    print('snap', s, 'MAE %.3f' % np.abs(pv[k]-y[isv][k]).mean())
