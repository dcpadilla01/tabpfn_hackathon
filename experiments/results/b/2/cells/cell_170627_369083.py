import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
extras = {}
e14 = load_saved('e014_base.parquet'); extras['gbm_pred']=e14[key+['gbm_pred']]
e9a = load_saved('e009_analog.parquet'); extras['knn'] = e9a[key+['knn28','binmean','log_knn','log_bin']]
e9s = load_saved('e009_spline2p.parquet'); extras['2p'] = e9s[key+['pred_2p','p_active','lvl_given_active','L_pred2p']]
D = F.merge(tt, on=key)
for k,v in extras.items(): D = D.merge(v, on=key, how='left')
feat0 = [c for c in F.columns if c not in key+['gbm_pred']]
extra_cols = {'gbm_pred':['gbm_pred'],'knn':['knn28','binmean','log_knn','log_bin'],'2p':['pred_2p','p_active','lvl_given_active','L_pred2p']}

ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]
med0 = ptr[feat0].astype(float).median()
def ridge_fit(Xtr, ytr, Xva, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym=ytr.mean()
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@(ytr-ym))
    return Zv@w+ym
Xtr0 = ptr[feat0].astype(float).fillna(med0).values; ytr=ptr[TARGET].values
Xpv0 = pv[feat0].astype(float).fillna(med0).values; ypv=pv[TARGET].values
base_p = ridge_fit(Xtr0,ytr,Xpv0,100.0)
print('V0 base proxy MAE:', round(np.abs(base_p-ypv).mean(),3))
resid = ypv-base_p
for k,cols in extra_cols.items():
    med = ptr[cols].astype(float).median()
    Xtr = ptr[feat0+cols].astype(float).fillna(0).values; Xtr[:, :len(feat0)] = np.where(np.isnan(Xtr[:, :len(feat0)]), med0.values, Xtr[:, :len(feat0)])
    Xpv = pv[feat0+cols].astype(float).fillna(0).values; Xpv[:, :len(feat0)] = np.where(np.isnan(Xpv[:, :len(feat0)]), med0.values, Xpv[:, :len(feat0)])
    p = ridge_fit(Xtr,ytr,Xpv,100.0)
    print(f'V+{k}: MAE {np.abs(p-ypv).mean():.3f}  resid-corr {pd.Series(resid).corr(pd.Series(pv[cols[0]].astype(float).fillna(med[cols[0]]).values)) if len(cols)==1 else ""}')
# winsorize variant
q = np.quantile(Xtr0, 0.995, axis=0)
Xtr_w = np.minimum(Xtr0, q); Xpv_w = np.minimum(Xpv0, q)
p = ridge_fit(Xtr_w,ytr,Xpv_w,100.0)
print('V0 winsor99.5 MAE:', round(np.abs(p-ypv).mean(),3))
q9 = np.quantile(Xtr0, 0.99, axis=0)
p = ridge_fit(np.minimum(Xtr0,q9),ytr,np.minimum(Xpv0,q9),100.0)
print('V0 winsor99 MAE:', round(np.abs(p-ypv).mean(),3))
# all extras + winsor
allcols = feat0+extra_cols['gbm_pred']+extra_cols['knn']+extra_cols['2p']
medA = ptr[allcols].astype(float).median()
XtrA = ptr[allcols].astype(float).fillna(medA).values; XpvA = pv[allcols].astype(float).fillna(medA).values
p = ridge_fit(XtrA,ytr,XpvA,100.0); print('V_all MAE:', round(np.abs(p-ypv).mean(),3))
qa = np.quantile(XtrA,0.995,axis=0)
p = ridge_fit(np.minimum(XtrA,qa),ytr,np.minimum(XpvA,qa),100.0); print('V_all winsor MAE:', round(np.abs(p-ypv).mean(),3))
# residual correlations with extras
for c in ['gbm_pred','knn28','binmean','pred_2p','p_active','lvl_given_active']:
    x = pv[c].astype(float).fillna(pv[c].astype(float).median()).values
    print(f'resid-corr {c}: {pd.Series(resid).corr(pd.Series(x)):.3f}')