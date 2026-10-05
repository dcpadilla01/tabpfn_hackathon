import pandas as pd, numpy as np
import agent_api

e = agent_api.load_saved('e011_table.parquet')
print('e011_table shape', e.shape)
cols = list(e.columns)
print('COLS:', ', '.join(cols))
print()

for name in ['selfcal_v1','churn_vol_v1','season_demo_v1','rfm_cadence_v1','rfm_traj_v1','ewma_block_v1','mkt_v1','comp_v1','demo_v1','rfm_v1']:
    try:
        d = agent_api.load_saved(name+'.parquet')
        cc = [c for c in d.columns if c not in ('household_key','snapshot_day')]
        print(name, d.shape, '|', ', '.join(cc[:80]))
    except Exception as ex:
        print(name, 'ERR', ex)
print()

tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share overall', float((tt.future_spend_4w==0).mean()))
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median'])
g['zero'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda s: float((s==0).mean()))
print(g)

def ridge_eval(df, cols, holdouts=(403,431), lam=30.0):
    cols = list(cols)
    d = tt.merge(df[['household_key','snapshot_day']+cols], on=['household_key','snapshot_day'], how='inner')
    X = d[cols].apply(pd.to_numeric, errors='coerce').values.astype(float)
    y = d['future_spend_4w'].values.astype(float)
    days = d['snapshot_day'].values
    out=[]
    for h in holdouts:
        tr = days!=h; te = days==h
        mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0)
        sd[~np.isfinite(sd)|(sd==0)]=1.0
        A = np.where(np.isnan(X[tr]),0.0,(X[tr]-mu)/sd)
        B = np.where(np.isnan(X[te]),0.0,(X[te]-mu)/sd)
        w = np.linalg.solve(A.T@A + lam*np.eye(len(cols)), A.T@y[tr])
        p = B@w
        out.append(float(np.mean(np.abs(p-y[te]))))
    return float(np.mean(out)), out

num_cols = [c for c in cols if c not in ('household_key','snapshot_day')]
print()
print('ridge all e011:', ridge_eval(e, num_cols))
