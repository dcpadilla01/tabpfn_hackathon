import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
e003 = agent_api.load_saved('e003_product_mix.parquet')
e002 = agent_api.load_saved('e002_marketing_v2.parquet')
eanalog = agent_api.load_saved('e009_analog.parquet')
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']

def proxy_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    p = B@w
    return np.abs(p-yva).mean()

c03 = e008.merge(e003.drop(columns=[c for c in e003.columns if c in e008.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
c02 = e008.merge(e002.drop(columns=[c for c in e002.columns if c in e008.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
call = c03.merge(eanalog.drop(columns=['knn28','binmean','log_knn','log_bin']), on=['household_key','snapshot_day'], how='inner') if False else c03

print('shapes: e008 %s e003 %s e002 %s' % (e008.shape, e003.shape, e002.shape))
print('proxy E008            v431: %.3f | v403+431: %.3f' % (proxy_mae(e008,[431]), proxy_mae(e008,[403,431])))
print('proxy E008+E003 mix   v431: %.3f | v403+431: %.3f' % (proxy_mae(c03,[431]), proxy_mae(c03,[403,431])))
print('proxy E008+E002 mkt   v431: %.3f | v403+431: %.3f' % (proxy_mae(c02,[431]), proxy_mae(c02,[403,431])))
c023 = c03.merge(e002.drop(columns=[c for c in e002.columns if c in c03.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
print('proxy E008+E003+E002  v431: %.3f | v403+431: %.3f' % (proxy_mae(c023,[431]), proxy_mae(c023,[403,431])))
# alpha sensitivity on best combo
for a in [0.3, 3.0, 30.0, 300.0]:
    print('proxy E008+E003+E002 alpha=%g v431: %.3f' % (a, proxy_mae(c023,[431],alpha=a)))
print('\ncols in e003 not in e008:', [c for c in e003.columns if c not in e008.columns][:40])
print('cols in e002 not in e008:', [c for c in e002.columns if c not in e008.columns][:40])