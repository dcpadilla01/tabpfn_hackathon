import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
trd = snapshot_days()['train']
feat0 = [c for c in F.columns if c not in key+['gbm_pred']]
def ridge_fit(Xtr, ytr, Xva, alpha=100.0):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym=ytr.mean()
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@(ytr-ym))
    return Zv@w+ym
def build_X(df, cols, med): return df[cols].astype(float).fillna(med).values
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]
ytr=ptr[TARGET].values; ypv=pv[TARGET].values
med0 = ptr[feat0].astype(float).median()
Xtr0 = build_X(ptr, feat0, med0); Xpv0 = build_X(pv, feat0, med0)
print('BASE:', round(np.abs(ridge_fit(Xtr0,ytr,Xpv0)-ypv).mean(),3))

# drift features (from ptr only for mapping consistency)
mkt28 = ptr.groupby('snapshot_day')['spend_28'].mean()
mkt28v = pv['spend_28'].mean()  # market at 431 from its own observable data
dr = pd.DataFrame({'mkt28': ptr['snapshot_day'].map(mkt28), 'drift28': ptr['spend_28'].astype(float)/ptr['snapshot_day'].map(mkt28)}, index=ptr.index)
drv = pd.DataFrame({'mkt28': mkt28v, 'drift28': pv['spend_28'].astype(float)/mkt28v}, index=pv.index)
XtrD = np.hstack([Xtr0, dr.values]); XpvD = np.hstack([Xpv0, drv.values])
print('+drift:', round(np.abs(ridge_fit(XtrD,ytr,XpvD)-ypv).mean(),3))

# phase fixed
g84 = ptr['gap_mean_84'].astype(float).replace(0,np.nan)
ph_tr = pd.DataFrame({'phase': ptr['recency'].astype(float)/(g84+1), 'trips_due': 28.0/(g84+1)}).replace([np.inf,-np.inf],np.nan)
g84v = pv['gap_mean_84'].astype(float).replace(0,np.nan)
ph_pv = pd.DataFrame({'phase': pv['recency'].astype(float)/(g84v+1), 'trips_due': 28.0/(g84v+1)}).replace([np.inf,-np.inf],np.nan)
mph = ph_tr.median()
XtrP = np.hstack([Xtr0, ph_tr.fillna(mph).values]); XpvP = np.hstack([Xpv0, ph_pv.fillna(mph).values])
print('+phase:', round(np.abs(ridge_fit(XtrP,ytr,XpvP)-ypv).mean(),3))

# log-target main fit
p_log = np.expm1(ridge_fit(Xtr0, np.log1p(ytr), Xpv0))
print('log-target:', round(np.abs(p_log-ypv).mean(),3))
p_logD = np.expm1(ridge_fit(XtrD, np.log1p(ytr), XpvD))
print('log-target+drift:', round(np.abs(p_logD-ypv).mean(),3))

# recency-weighted ridge (newer snapshots weigh more)
def wridge_fit(Xtr, ytr, wt, Xva, alpha=100.0):
    mu = (Xtr*wt[:,None]).sum(0)/wt.sum(); sg = np.sqrt(((Xtr-mu)**2*wt[:,None]).sum(0)/wt.sum())+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym=(ytr*wt).sum()/wt.sum()
    ZtW = Z*wt[:,None]
    w = np.linalg.solve(Z.T@ZtW+alpha*np.eye(Z.shape[1]), ZtW.T@(ytr-ym))
    return Zv@w+ym
snap_of_tr = ptr['snapshot_day'].values
for hl in [1e9, 300, 150]:
    wt = 0.5**((431-snap_of_tr)/hl) if hl<1e8 else np.ones(len(ptr))
    p = wridge_fit(Xtr0, ytr, wt, Xpv0)
    print(f'weighted hl={hl}:', round(np.abs(p-ypv).mean(),3))

# stacked OOF (clean)
snaps = sorted(trd)
folds = [snaps[0:3], snaps[3:6], snaps[6:9], snaps[9:12], snaps[12:13]]
oof = pd.Series(np.nan, index=D.index)
ylog_all = np.log1p(D[TARGET].astype(float))
for f in folds:
    trn = D[D.snapshot_day.isin([s for s in snaps if s not in f])]; tst = D[D.snapshot_day.isin(f)]
    mtr = trn[feat0].astype(float).median()
    oof.loc[tst.index] = np.expm1(ridge_fit(build_X(trn,feat0,mtr), np.log1p(trn[TARGET].astype(float)).values, build_X(tst,feat0,mtr)))
trn_all = D[D.snapshot_day.isin(snaps)]; pv_rows = D[~D.snapshot_day.isin(snaps)]
mtrA = trn_all[feat0].astype(float).median()
oof.loc[pv_rows.index] = np.expm1(ridge_fit(build_X(trn_all,feat0,mtrA), np.log1p(trn_all[TARGET].astype(float)).values, build_X(pv_rows,feat0,mtrA)))
D['stk'] = oof
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]
print('stk alone @431:', round(np.abs(pv['stk'].astype(float).values-ypv).mean(),2), '| corr:', round(pv['stk'].astype(float).corr(pv[TARGET]),3))
mstk = ptr['stk'].astype(float).median()
XtrS = np.hstack([Xtr0, ptr[['stk']].astype(float).fillna(mstk).values]); XpvS = np.hstack([Xpv0, pv[['stk']].astype(float).fillna(mstk).values])
print('+stacked:', round(np.abs(ridge_fit(XtrS,ytr,XpvS)-ypv).mean(),3))
XtrSD = np.hstack([XtrD, ptr[['stk']].astype(float).fillna(mstk).values]); XpvSD = np.hstack([XpvD, pv[['stk']].astype(float).fillna(mstk).values])
print('+stacked+drift:', round(np.abs(ridge_fit(XtrSD,ytr,XpvSD)-ypv).mean(),3))
p_logS = np.expm1(ridge_fit(XtrS, np.log1p(ytr), XpvS))
print('log-target+stacked:', round(np.abs(p_logS-ypv).mean(),3))