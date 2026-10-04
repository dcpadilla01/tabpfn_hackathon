import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

t = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
days = agent_api.snapshot_days()
tr_days = days['train']

df = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]

def ridge_fit_pred(Xtr, ytr, Xva, alpha=1.0, standardize=True):
    Xtr = np.asarray(Xtr, float); Xva = np.asarray(Xva, float)
    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
        Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
    Xtr1 = np.hstack([Xtr, np.ones((len(Xtr),1))]); Xva1 = np.hstack([Xva, np.ones((len(Xva),1))])
    A = Xtr1.T@Xtr1 + alpha*np.eye(Xtr1.shape[1]); A[-1,-1]-=alpha
    w = np.linalg.solve(A, Xtr1.T@ytr)
    return Xva1@w, w

def proxy_mae(table, feat_subset=None, train_days=None, val_days=None, alpha=1.0, impute='zero'):
    d = table if feat_subset is None else table[['household_key','snapshot_day']+list(feat_subset)]
    f = [c for c in d.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(d, on=['household_key','snapshot_day'], how='inner')
    train_days = train_days or [x for x in tr_days if x not in val_days] if val_days else tr_days
    tr = m[m.snapshot_day.isin(train_days)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).copy(); Xva = va[f].astype(float).copy()
    if impute=='zero':
        Xtr=Xtr.fillna(0.0); Xva=Xva.fillna(0.0)
    else:
        med=Xtr.median(); Xtr=Xtr.fillna(med); Xva=Xva.fillna(med)
    p,_ = ridge_fit_pred(Xtr.values, tr['future_spend_4w'].values, Xva.values, alpha=alpha)
    return np.abs(p - va['future_spend_4w'].values).mean()

# proxy split: train on 95..403, validate on 431 (mimics future-shift)
for alpha in [0.1, 1.0, 10.0, 100.0]:
    m = proxy_mae(t, val_days=[431], alpha=alpha)
    print(f'proxy val=431 alpha={alpha}: MAE {m:.3f}')
# also val on {403,431}
for alpha in [1.0, 10.0]:
    m = proxy_mae(t, val_days=[403,431], alpha=alpha)
    print(f'proxy val=403+431 alpha={alpha}: MAE {m:.3f}')

# residual pattern by snapshot day (in-sample-ish, train on 95..403 predict 431 and also per-day on train)
m = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
f = feats
tr = m[m.snapshot_day.isin([x for x in tr_days if x!=431])]
va = m[m.snapshot_day==431]
Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
p,_ = ridge_fit_pred(Xtr, tr['future_spend_4w'].values, Xva, alpha=1.0)
res = p - va['future_spend_4w'].values
print('\nday-431 proxy: MAE %.3f  mean pred %.1f  mean actual %.1f  bias %.2f' % (np.abs(res).mean(), p.mean(), va['future_spend_4w'].mean(), res.mean()))
# error by target decile
q = pd.qcut(va['future_spend_4w'].values, 10, duplicates='drop')
print(pd.DataFrame({'y':va['future_spend_4w'].values,'p':p,'q':q}).groupby('q', observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g.y.mean(),'p':g.p.mean(),'mae':np.abs(g.p-g.y).mean()})).round(1))