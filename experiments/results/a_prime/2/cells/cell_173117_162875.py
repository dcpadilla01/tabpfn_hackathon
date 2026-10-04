import agent_api, pandas as pd, numpy as np
base = agent_api.load_saved('e009_ewma_longlags.parquet')
m2 = agent_api.load_saved('e017_season_hazard.parquet')
tt = agent_api.train_targets()
def ev(df, alpha=100.0):
    df = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
    cols=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()
b = ev(base)
haz = m2.drop(columns=['hh_season_next4','hh_season_next4_x','glob_season_next4'])
rh = ev(haz); print('hazard-only (4 feats): %.2f / %.2f  d %+.2f'%(rh[0],rh[1],rh[1]-b[1]))
r2 = ev(m2); print('full v2 (7 feats):     %.2f / %.2f  d %+.2f'%(r2[0],r2[1],r2[1]-b[1]))
p = agent_api.save_table(haz, 'e017_hazard_only.parquet'); print('saved', p)