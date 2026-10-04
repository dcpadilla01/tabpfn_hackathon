import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
trd = snapshot_days()['train']
feat0 = [c for c in F.columns if c not in key+['gbm_pred']]
# skewness check
sk = D[feat0].astype(float).skew().abs().sort_values(ascending=False)
print('most skewed:', list(sk.index[:10].values), np.round(sk.values[:10],1))

ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]
def ridge_fit(Xtr, ytr, Xva, alpha=100.0):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym=ytr.mean()
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@(ytr-ym))
    return Zv@w+ym
def build_X(df, cols, med):
    return df[cols].astype(float).fillna(med).values
med0 = ptr[feat0].astype(float).median()
Xtr0 = build_X(ptr, feat0, med0); Xpv0 = build_X(pv, feat0, med0); ytr=ptr[TARGET].values; ypv=pv[TARGET].values
base = ridge_fit(Xtr0,ytr,Xpv0)
print('BASE proxy 431 MAE:', round(np.abs(base-ypv).mean(),3))

# --- 1. drop families ---
mkt = ['n_campaigns_targeted','tA','tB','tC','targeted','active_campaign','days_since_camp_start','n_redeem_life','n_redeem_84','days_since_redeem']
dept = ['sh_grocer','sh_drug g','sh_produc','sh_cosmet','sh_nutrit','sh_meat','sh_deli','sh_pastry']
misc = ['disc_share_28','private_share_84','x_disc_share_84']
drop1 = [c for c in mkt+dept+misc if c in feat0]
f1 = [c for c in feat0 if c not in drop1]
m1 = ptr[f1].astype(float).median()
p = ridge_fit(build_X(ptr,f1,m1),ytr,build_X(pv,f1,m1))
print('drop mkt/dept/disc (%d feats):'%len(f1), round(np.abs(p-ypv).mean(),3))

# --- 2. shrinkage features ---
gmed = ptr[TARGET].median()
for k in [4,8,16]:
    w = ptr['trips_84'].astype(float)/(ptr['trips_84'].astype(float)+k)
    shr = w*ptr['spend_28'].astype(float) + (1-w)*gmed
    ptr2 = ptr.assign(shr=shr); pv2 = pv.assign(shr=(pv['trips_84'].astype(float)/(pv['trips_84'].astype(float)+k))*pv['spend_28'].astype(float)+(1-w.mean()*0+1)*gmed*0 + (1-(pv['trips_84'].astype(float)/(pv['trips_84'].astype(float)+k)))*gmed)
    Xtr = np.hstack([Xtr0, ptr2[['shr']].astype(float).values]); Xpv = np.hstack([Xpv0, pv2[['shr']].astype(float).values])
    p = ridge_fit(Xtr,ytr,Xpv)
    print(f'+shrink k={k}:', round(np.abs(p-ypv).mean(),3), '| shr alone MAE:', round(np.abs(ptr2.shr.values-pv2.shr.assign().values*0 - ypv + pv2.shr.values - ypv).mean(),1) if False else round(np.abs(pv2.shr.values-ypv).mean(),1))

# --- 3. phase features ---
ph = pd.DataFrame({
  'phase': ptr['recency'].astype(float)/(ptr['gap_mean_84'].astype(float)+1),
  'trips_due': 28.0/(ptr['gap_mean_84'].astype(float)+1),
}, index=ptr.index)
phv = pd.DataFrame({
  'phase': pv['recency'].astype(float)/(pv['gap_mean_84'].astype(float)+1),
  'trips_due': 28.0/(pv['gap_mean_84'].astype(float)+1),
}, index=pv.index)
Xtr = np.hstack([Xtr0, ph.values]); Xpv = np.hstack([Xpv0, phv.values])
p = ridge_fit(Xtr,ytr,Xpv); print('+phase:', round(np.abs(p-ypv).mean(),3))

# --- 4. drift/market features ---
mkt28 = D.groupby('snapshot_day')['spend_28'].mean()
glob28 = ptr['spend_28'].mean()
dr = pd.DataFrame({'mkt28': ptr['snapshot_day'].map(mkt28), 'drift28': ptr['spend_28'].astype(float)/ptr['snapshot_day'].map(mkt28)}, index=ptr.index)
drv = pd.DataFrame({'mkt28': pv['snapshot_day'].map(mkt28), 'drift28': pv['spend_28'].astype(float)/pv['snapshot_day'].map(mkt28)}, index=pv.index)
Xtr = np.hstack([Xtr0, dr.values]); Xpv = np.hstack([Xpv0, drv.values])
p = ridge_fit(Xtr,ytr,Xpv); print('+market drift:', round(np.abs(p-ypv).mean(),3))

# --- 5. log/clip extremes ---
ext = ['c_dev84','c_dev28','ratio_lag','qty28','last_bqty','x_b_mean_qty','spend_life','spend_364','x_life_total']
ext = [c for c in ext if c in feat0]
XtrE = Xtr0.copy(); XpvE = Xpv0.copy()
for c in ext:
    i = feat0.index(c)
    XtrE[:,i] = np.log1p(np.maximum(Xtr0[:,i],0)); XpvE[:,i] = np.log1p(np.maximum(Xpv0[:,i],0))
p = ridge_fit(XtrE,ytr,XpvE); print('+log extremes:', round(np.abs(p-ypv).mean(),3))

# --- 6. stacked OOF ridge-on-log feature ---
snaps = sorted(D[D.snapshot_day.isin(trd)].snapshot_day.unique())
folds = [snaps[0:3], snaps[3:6], snaps[6:9], snaps[9:12], snaps[12:13]]
oof = pd.Series(np.nan, index=D.index)
ylog = np.log1p(D[TARGET].astype(float))
for f in folds:
    trn = D[D.snapshot_day.isin([s for s in snaps if s not in f])]
    tst = D[D.snapshot_day.isin(f)]
    mtr = trn[feat0].astype(float).median()
    plog = ridge_fit(build_X(trn,feat0,mtr), np.log1p(trn[TARGET].astype(float)).values, build_X(tst,feat0,mtr))
    oof.loc[tst.index] = np.expm1(plog)
# val rows: fit on all train snaps
trn_all = D[D.snapshot_day.isin(snaps)]
mtrA = trn_all[feat0].astype(float).median()
pv_rows = D[~D.snapshot_day.isin(snaps)]
plogV = ridge_fit(build_X(trn_all,feat0,mtrA), np.log1p(trn_all[TARGET].astype(float)).values, build_X(pv_rows,feat0,mtrA))
oof.loc[pv_rows.index] = np.expm1(plogV)
D['stk'] = oof
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]
Xtr0 = build_X(ptr, feat0, med0); Xpv0 = build_X(pv, feat0, med0)
print('stacked alone MAE@431:', round(np.abs(ptr['stk'].values*0+pv['stk'].values-ypv).mean(),2))
Xtr = np.hstack([Xtr0, ptr[['stk']].astype(float).values]); Xpv = np.hstack([Xpv0, pv[['stk']].astype(float).values])
p = ridge_fit(Xtr,ytr,Xpv); print('+stacked OOF:', round(np.abs(p-ypv).mean(),3))
Xtr = np.hstack([XtrE, ptr[['stk']].astype(float).values]); Xpv = np.hstack([XpvE, pv[['stk']].astype(float).values])
p = ridge_fit(Xtr,ytr,Xpv); print('+stacked OOF +logext:', round(np.abs(p-ypv).mean(),3))
print('corr(stk,y)@431:', round(pv['stk'].astype(float).corr(pv[TARGET]),3), '| corr(base_pred, y)@431:', round(pd.Series(base).corr(pd.Series(ypv)),3))