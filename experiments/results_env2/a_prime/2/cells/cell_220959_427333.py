import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
feats = [c for c in df.columns if c not in ('household_key','snapshot_day')]
d2 = df.merge(tt, on=['household_key','snapshot_day'], how='left')
sd = agent_api.snapshot_days(); tr, va = sd['train'], sd['validation']
mtr = d2.snapshot_day.isin(tr); mva = d2.snapshot_day.isin(va)
ytr = d2.loc[mtr,'future_spend_4w'].values; yva = d2.loc[mva,'future_spend_4w'].values
print('ytr nan:', np.isnan(ytr).sum(), 'yva nan:', np.isnan(yva).sum(), 'ntr', mtr.sum(), 'nva', mva.sum())

def fit_eval(Xtr, ytr_, Xva, yva_, alphas=(1,10,100,300,1000,3000)):
    mu = np.nanmean(Xtr, axis=0); sg = np.nanstd(Xtr, axis=0)+1e-9
    Ztr = np.nan_to_num((Xtr-mu)/sg); Zva = np.nan_to_num((Xva-mu)/sg)
    best=None
    for a in alphas:
        w = np.linalg.solve(Ztr.T@Ztr + a*np.eye(Ztr.shape[1]), Ztr.T@(ytr_-ytr_.mean()))
        pred = Zva@w + ytr_.mean()
        mae = np.abs(pred-yva_).mean()
        if best is None or mae<best[0]: best=(mae,a)
    return best

Xraw = d2[feats].values.astype(float)
print('FULL RAW   MAE %.3f (a %s)' % fit_eval(Xraw[mtr.values], ytr, Xraw[mva.values], yva))
Xl = d2[feats].astype(float).copy()
logcols = [c for c in feats if any(c.startswith(p) for p in ('spend','ew_','avg_basket','basket_','longrun_wk','n_products','n_stores'))]
for c in logcols: Xl[c]=np.log1p(Xl[c].clip(lower=0))
print('FULL LOG   MAE %.3f (a %s)' % fit_eval(Xl.values[mtr.values], ytr, Xl.values[mva.values], yva))
mu=np.nanmean(Xl.values[mtr.values],axis=0); sg=np.nanstd(Xl.values[mtr.values],axis=0)+1e-9
Ztr=np.nan_to_num((Xl.values[mtr.values]-mu)/sg); Zva=np.nan_to_num((Xl.values[mva.values]-mu)/sg)
ylog=np.log1p(ytr); best=None
for a in [1,10,100,300,1000]:
    w=np.linalg.solve(Ztr.T@Ztr+a*np.eye(Ztr.shape[1]), Ztr.T@(ylog-ylog.mean()))
    pred=np.expm1(np.clip(Zva@w+ylog.mean(),0,8))
    mae=np.abs(pred-yva).mean()
    if best is None or mae<best[0]: best=(mae,a)
print('FULL LOGX/LOGY expm1 MAE %.3f (a %s)' % best)