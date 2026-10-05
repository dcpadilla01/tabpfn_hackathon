import agent_api as A, pandas as pd, numpy as np, xgboost as xgb, time
t0=time.time()
print('xgb', xgb.__version__)
F = A.load_saved('allF.parquet'); W = A.load_saved('f_weekly.parquet')
ycol='future_spend_4w'
W2 = W.drop(columns=[ycol], errors='ignore')
F = F.merge(W2, on=['household_key','snapshot_day'], how='left')
print('merged', F.shape)
base_cols = [c for c in F.columns if c not in ('household_key','snapshot_day',ycol)]
week_cols = [c for c in W2.columns if c not in ('household_key','snapshot_day')]
print('n base', len(base_cols), 'n week', len(week_cols))
print(F[week_cols].isna().mean().round(3).to_string())
tr_days=[95,123,151,179,207,235,263,291,319,347,375,403,431]
Ftr = F[F.snapshot_day.isin(tr_days) & F[ycol].notna()].copy()
print('train rows', Ftr.shape, 'time', round(time.time()-t0,1))
# quick correlation of weekly feats with target
sub = Ftr[week_cols+['snapshot_day']].copy(); sub['y']=Ftr[ycol].values
cor = sub.corr()['y'].drop('y').drop('snapshot_day')
print(cor.round(3).to_string())
print('elapsed', round(time.time()-t0,1))
