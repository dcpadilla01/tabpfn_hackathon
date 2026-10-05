import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
d2 = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in d2.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
y = d2.future_spend_4w.values
def fit_pred(Xtr, ytr, Xva, a=300):
    mu=np.nanmean(Xtr,axis=0); sg=np.nanstd(Xtr,axis=0)+1e-9
    Ztr=np.nan_to_num((Xtr-mu)/sg); Zva=np.nan_to_num((Xva-mu)/sg)
    w=np.linalg.solve(Ztr.T@Ztr+a*np.eye(Ztr.shape[1]), Ztr.T@(ytr-ytr.mean()))
    return Zva@w+ytr.mean()
mfit=(d2.snapshot_day<=375).values; miv=(d2.snapshot_day>=403).values
X0 = d2[feats].values.astype(float)
pred = fit_pred(X0[mfit], y[mfit], X0[miv], 300)
yv = y[miv]; err = np.abs(pred-yv)
act = d2.active_28.values[miv]
print('MAE inactive: %.2f (n=%d, mean y=%.1f, median y=%.1f, mean pred=%.1f)' % (err[~act].mean(), (~act).sum(), yv[~act].mean(), np.median(yv[~act]), pred[~act].mean()))
print('MAE active  : %.2f (n=%d, mean y=%.1f, median y=%.1f, mean pred=%.1f)' % (err[act].mean(), act.sum(), yv[act].mean(), np.median(yv[act]), pred[act].mean()))
for lo,hi in [(0,25),(25,75),(75,150),(150,300),(300,1e9)]:
    m=(yv>=lo)&(yv<hi)
    print('y in [%4d,%5d): n=%5d mean pred=%7.1f mean y=%7.1f MAE=%6.1f' % (lo,hi,m.sum(),pred[m].mean(),yv[m].mean(),err[m].mean()))
# group-mean predictors: what MAE would constant-per-group give?
for grp in [d2.active_28.values[miv], (d2.days_since_last.fillna(999).values[miv]>56)]:
    gm = pd.Series(yv).groupby(grp).transform('median').values
    print('group-median MAE: %.2f' % np.abs(gm-yv).mean())