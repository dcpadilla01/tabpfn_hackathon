import pandas as pd, numpy as np, agent_api
# offline prototype: aligned 4-week lag spends (no leakage: windows end at snapshot day)
v = agent_api.snapshot(459)
tx = v.transactions[['household_key','day','sales_value']]
hh = np.sort(tx.household_key.unique())
hidx = {h:i for i,h in enumerate(hh)}
spend = np.zeros((len(hh), 460), dtype=np.float64)
np.add.at(spend, (tx.household_key.map(hidx).values, tx.day.values), tx.sales_value.values)
cum = np.concatenate([np.zeros((len(hh),1)), spend.cumsum(1)], axis=1)  # cum[:,d] = spend days 1..d

df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
d2 = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
rows = d2.household_key.map(hidx).values; s = d2.snapshot_day.values
lag = {}
for k in range(1, 14):
    hi = np.clip(s-28*k, 0, 459); lo = np.clip(s-28*(k+1), 0, 459)
    lag['lag%d'%k] = cum[rows, hi] - cum[rows, lo]
L = pd.DataFrame(lag)
print(L.describe().round(1).T[['mean','50%','max']])

feats = [c for c in d2.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
y = d2.future_spend_4w.values
def fit_pred(Xtr, ytr, Xva, a=300):
    mu=np.nanmean(Xtr,axis=0); sg=np.nanstd(Xtr,axis=0)+1e-9
    Ztr=np.nan_to_num((Xtr-mu)/sg); Zva=np.nan_to_num((Xva-mu)/sg)
    w=np.linalg.solve(Ztr.T@Ztr+a*np.eye(Ztr.shape[1]), Ztr.T@(ytr-ytr.mean()))
    return Zva@w+ytr.mean()
mfit=(d2.snapshot_day<=375).values; miv=(d2.snapshot_day>=403).values
X0 = d2[feats].values.astype(float)
base = np.abs(fit_pred(X0[mfit], y[mfit], X0[miv], 300)-y[miv]).mean()
print('base %.3f' % base)
for ks in [[1],[1,2],[1,2,3],[1,2,3,4],[1,2,3,4,5,6],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,10,11,12,13]]:
    Xn = np.column_stack([X0]+[lag['lag%d'%k] for k in ks])
    m = np.abs(fit_pred(Xn[mfit], y[mfit], Xn[miv], 300)-y[miv]).mean()
    print('lags %-25s %.3f (%+.3f)' % (str(ks), m, m-base))
# household fixed effect: mean of available lags 1..8
Lm = np.column_stack([lag['lag%d'%k] for k in range(1,9)])
fe = np.nanmean(np.where(Lm>0, Lm, np.nan), axis=1)
Xn = np.column_stack([X0, np.nan_to_num(fe)])
m = np.abs(fit_pred(Xn[mfit], y[mfit], Xn[miv], 300)-y[miv]).mean()
print('fe_mean_lags %.3f (%+.3f)' % (m, m-base))