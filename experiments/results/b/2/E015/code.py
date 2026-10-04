import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets()
print('targets:', tt.shape, list(tt.columns))
print(tt[TARGET].describe().round(2))
for name in ['e013_denoise','e014_base','e014_gbm_oob','e014_stack']:
    try:
        df = load_saved(name + '.parquet')
        print('==', name, df.shape)
        print('dtypes:', dict(df.dtypes.value_counts()))
        print('first cols:', list(df.columns)[:14])
        print('last cols:', list(df.columns)[-6:])
        print('snapshots:', sorted(df['snapshot_day'].unique()))
        print('nan%:', round(df.isna().mean().mean()*100, 2))
    except Exception as e:
        print('==', name, 'ERR', repr(e))

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days
sd = snapshot_days(); trd = sd['train']; vld = sd['validation']
tt = train_targets(); key=['household_key','snapshot_day']
base = load_saved('e014_base.parquet'); oob = load_saved('e014_gbm_oob.parquet'); stk = load_saved('e014_stack.parquet')
cmp = base[key+['gbm_pred']].merge(oob[key+['gbm_corr']], on=key)
print('gbm_pred vs gbm_corr corr:', round(cmp['gbm_pred'].corr(cmp['gbm_corr']),4))
v = cmp[cmp.snapshot_day.isin(vld)]; t = cmp[cmp.snapshot_day.isin(trd)]
print('val: pred==corr?', np.allclose(v.gbm_pred, v.gbm_corr), '| train: pred==corr?', np.allclose(t.gbm_pred, t.gbm_corr))
m = oob[key+['gbm_corr']].merge(tt, on=key)
m['ae'] = (m['gbm_corr']-m[TARGET]).abs()
g = m.groupby('snapshot_day').apply(lambda d: pd.Series({'mae': d.ae.mean(), 'corr': d.gbm_corr.corr(d[TARGET]), 'n': len(d)}))
print(g.round(3))
print('TRAIN overall gbm_corr MAE:', round(m.ae.mean(),2))
print('val gbm_corr describe:'); print(v.gbm_corr.describe().round(2))
print('train gbm_corr describe:'); print(t.gbm_corr.describe().round(2))
for nm in ['e009_analog','e009_demo_mkt','e009_spline2p']:
    d = load_saved(nm+'.parquet'); print('==',nm, d.shape); print(list(d.columns))

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days
sd = snapshot_days(); trd = sd['train']
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
feat = [c for c in F.columns if c not in key+['gbm_pred']]
D = F.merge(tt, on=key)
print('n feat', len(feat))
# simple baselines on pseudo-val (snap 431)
pv = D[D.snapshot_day==431]; ptr = D[D.snapshot_day<431]
print('y mean/med pv:', pv[TARGET].mean().round(1), pv[TARGET].median().round(1), '| train:', ptr[TARGET].mean().round(1), ptr[TARGET].median().round(1))
for c in ['spend_28','ewma28_4w','fwd28_mean','pred_2p' ]:
    if c in D.columns:
        print(f'baseline {c}: pv MAE', round((pv[c]-pv[TARGET]).abs().mean(),2), '| scaled .9:', round((0.9*pv[c]-pv[TARGET]).abs().mean(),2))
print('baseline 0:', round(pv[TARGET].abs().mean(),2), 'median:', round((ptr[TARGET].median()-pv[TARGET]).abs().mean(),2))

def ridge_fit(Xtr, ytr, Xva, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z = (Xtr-mu)/sg; Zv=(Xva-mu)/sg
    A = Z.T@Z + alpha*np.eye(Z.shape[1]); b = Z.T@ytr
    w = np.linalg.solve(A,b)
    return Zv@w, w, mu, sg
Xtr = ptr[feat].astype(float).fillna(ptr[feat].astype(float).median()).values
ytr = ptr[TARGET].values; Xpv = pv[feat].astype(float).fillna(ptr[feat].astype(float).median()).values; ypv = pv[TARGET].values
# pick alpha on inner split (train<=375, val 403)
pi = D[D.snapshot_day==403]; pt = D[D.snapshot_day<=375]
Xt2 = pt[feat].astype(float).fillna(ptr[feat].astype(float).median()).values; yt2=pt[TARGET].values
Xi2 = pi[feat].astype(float).fillna(ptr[feat].astype(float).median()).values; yi2=pi[TARGET].values
best=None
for a in [1,3,10,30,100,300,1000]:
    p,_,_,_ = ridge_fit(Xt2, yt2, Xi2, a)
    m = np.abs(p-yi2).mean()
    if best is None or m<best[1]: best=(a,m)
print('inner best alpha, MAE@403:', best)
p, w, mu, sg = ridge_fit(Xtr, ytr, Xpv, best[0])
print('pseudo-val 431 ridge MAE:', round(np.abs(p-ypv).mean(),2), 'bias:', round((p-ypv).mean(),2))
# per-snapshot residual bias on train fit
for s in [95,207,319,403]:
    m_ = D[D.snapshot_day==s]; Xs = m_[feat].astype(float).fillna(ptr[feat].astype(float).median()).values
    ps = (Xs-mu)/sg@w
    print(f'  snap {s}: bias {np.round((ps-m_[TARGET]).mean(),2)} MAE {np.round(np.abs(ps-m_[TARGET]).mean(),2)}')
# gbm_pred as feature?
if 'gbm_pred' in F.columns:
    g = F[key+['gbm_pred']].merge(tt, on=key)
    print('corr(gbm_pred,y) train:', round(g.gbm_pred.corr(g[TARGET]),3))
    Xtr2 = np.hstack([Xtr, ptr[['gbm_pred']].astype(float).values]); Xpv2 = np.hstack([Xpv, pv[['gbm_pred']].astype(float).values])
    p2,_,_,_ = ridge_fit(Xtr2, ytr, Xpv2, best[0])
    print('pseudo-val MAE with gbm_pred:', round(np.abs(p2-ypv).mean(),2))

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
feat = [c for c in F.columns if c not in key+['gbm_pred']]
D = F.merge(tt, on=key)
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]
Xtr = ptr[feat].astype(float)
print('ytr mean:', ptr[TARGET].mean().round(2), 'nan y:', ptr[TARGET].isna().sum())
inf_ct = np.isinf(Xtr.values).sum(axis=0)
bad = [(feat[i], int(c)) for i,c in enumerate(inf_ct) if c>0]
print('cols with inf:', bad[:20], 'total inf cols:', len(bad))
nan_ct = Xtr.isna().sum(); print('cols all-NaN in ptr:', [feat[i] for i,c in enumerate(nan_ct) if c==len(ptr)][:20])
print('max abs values:', Xtr.abs().max().sort_values().iloc[-8:])
print('const cols in ptr:', [c for c in feat if Xtr[c].nunique(dropna=True)<=1][:20])

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]

def ridge_fit(Xtr, ytr, Xva, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@ytr)
    return Zv@w
# minimal test: spend_28 + spend_84
for cols in [['spend_28'],['spend_28','spend_84'],['spend_28','spend_84','spend_364']]:
    Xtr = ptr[cols].astype(float).values; Xpv = pv[cols].astype(float).values
    p = ridge_fit(Xtr, ptr[TARGET].values, Xpv, 1.0)
    print(cols, 'pv MAE', round(np.abs(p-pv[TARGET]).values.mean(),2), 'pred mean', round(p.mean(),1), 'coef', np.round(np.linalg.solve((Xtr-Xtr.mean(0))/Xtr.std(0).T@((Xtr-Xtr.mean(0))/Xtr.std(0))+np.eye(len(cols)), ((Xtr-Xtr.mean(0))/Xtr.std(0)).T@ptr[TARGET].values),3))
# full feat in-sample check
feat = [c for c in F.columns if c not in key+['gbm_pred']]
med = ptr[feat].astype(float).median()
Xtr = ptr[feat].astype(float).fillna(med).values; ytr=ptr[TARGET].values
p_in = ridge_fit(Xtr, ytr, Xtr, 1.0)
print('IN-SAMPLE full ridge MAE:', round(np.abs(p_in-ytr).mean(),2), 'pred mean:', round(p_in.mean(),1))
# check std of standardized design
Z=(Xtr-Xtr.mean(0))/Xtr.std(0)
print('Z std range:', np.round(Z.std(0).min(),4), np.round(Z.std(0).max(),4))
print('cond num of ZtZ:', np.round(np.linalg.cond(Z.T@Z),1))

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]

def ridge_fit(Xtr, ytr, Xva, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@ytr)
    return Zv@w, w
for cols in [['spend_28'],['spend_28','spend_84'],['spend_28','spend_84','spend_364']]:
    Xtr = ptr[cols].astype(float).values; Xpv = pv[cols].astype(float).values
    p, w = ridge_fit(Xtr, ptr[TARGET].values, Xpv, 1.0)
    print(cols, 'pv MAE', round(np.abs(p-pv[TARGET]).values.mean(),2), 'pred mean', round(p.mean(),1), 'coef', np.round(w,3))
feat = [c for c in F.columns if c not in key+['gbm_pred']]
med = ptr[feat].astype(float).median()
Xtr = ptr[feat].astype(float).fillna(med).values; ytr=ptr[TARGET].values
p_in, _ = ridge_fit(Xtr, ytr, Xtr, 1.0)
print('IN-SAMPLE full ridge MAE:', round(np.abs(p_in-ytr).mean(),2), 'pred mean:', round(p_in.mean(),1), 'pred std:', round(p_in.std(),1))
Z=(Xtr-Xtr.mean(0))/(Xtr.std(0)+1e-9)
print('Z std min:', round(Z.std(0).min(),6), 'cond:', round(np.linalg.cond(Z.T@Z),1))
# per-snapshot means of key features
for c in ['spend_28','fwd28_mean','wk0']:
    print(c, 'ptr mean', round(ptr[c].mean(),1), 'pv mean', round(pv[c].mean(),1))

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]

def ridge_fit(Xtr, ytr, Xva, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym = ytr.mean(); yc = ytr-ym
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@yc)
    return Zv@w + ym
feat = [c for c in F.columns if c not in key+['gbm_pred']]
med = ptr[feat].astype(float).median()
Xtr = ptr[feat].astype(float).fillna(med).values; ytr=ptr[TARGET].values
Xpv = pv[feat].astype(float).fillna(med).values; ypv=pv[TARGET].values
p_in,_ = ridge_fit(Xtr,ytr,Xtr,1.0)
print('in-sample MAE:', round(np.abs(p_in-ytr).mean(),2), 'pred mean', round(p_in.mean(),1), 'std', round(p_in.std(),1))
# inner split for alpha: train <=375, val 403
pi = D[D.snapshot_day==403]; pt = D[D.snapshot_day<=375]
med2 = pt[feat].astype(float).median()
Xt2 = pt[feat].astype(float).fillna(med2).values; yt2=pt[TARGET].values
Xi2 = pi[feat].astype(float).fillna(med2).values; yi2=pi[TARGET].values
for a in [1,3,10,30,100]:
    p,_ = ridge_fit(Xt2,yt2,Xi2,a); print('alpha',a,'MAE@403', round(np.abs(p-yi2).mean(),2))
best_a = 10
p, w = ridge_fit(Xtr,ytr,Xpv,best_a)
print('PSEUDO-VAL 431 ridge MAE:', round(np.abs(p-ypv).mean(),2), 'bias:', round((p-ypv).mean(),2), 'pred std:', round(p.std(),1), 'y std:', round(ypv.std(),1))
for s in [95,207,319,403]:
    m_ = D[D.snapshot_day==s]; Xs = m_[feat].astype(float).fillna(med).values
    ps = ridge_fit(Xtr,ytr,Xs,best_a)
    print(f'  snap {s}: bias {np.round((ps-m_[TARGET]).mean(),2)} MAE {np.round(np.abs(ps-m_[TARGET]).mean(),2)}')
print('corr(pred,y) pv:', round(pd.Series(p).corr(pd.Series(ypv)),3))
# gbm_pred as extra feature
Xtr2 = np.hstack([Xtr, ptr[['gbm_pred']].astype(float).values]); Xpv2 = np.hstack([Xpv, pv[['gbm_pred']].astype(float).values])
p2,_ = ridge_fit(Xtr2,ytr,Xpv2,best_a)
print('pv MAE + gbm_pred:', round(np.abs(p2-ypv).mean(),2))

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
ptr = D[D.snapshot_day<431]; pv = D[D.snapshot_day==431]

def ridge_fit(Xtr, ytr, Xva, alpha):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym = ytr.mean(); yc = ytr-ym
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@yc)
    return Zv@w + ym
feat = [c for c in F.columns if c not in key+['gbm_pred']]
med = ptr[feat].astype(float).median()
Xtr = ptr[feat].astype(float).fillna(med).values; ytr=ptr[TARGET].values
Xpv = pv[feat].astype(float).fillna(med).values; ypv=pv[TARGET].values
p_in = ridge_fit(Xtr,ytr,Xtr,1.0)
print('in-sample MAE:', round(np.abs(p_in-ytr).mean(),2), 'pred mean', round(p_in.mean(),1), 'std', round(p_in.std(),1))
pi = D[D.snapshot_day==403]; pt = D[D.snapshot_day<=375]
med2 = pt[feat].astype(float).median()
Xt2 = pt[feat].astype(float).fillna(med2).values; yt2=pt[TARGET].values
Xi2 = pi[feat].astype(float).fillna(med2).values; yi2=pi[TARGET].values
for a in [1,3,10,30,100]:
    p = ridge_fit(Xt2,yt2,Xi2,a); print('alpha',a,'MAE@403', round(np.abs(p-yi2).mean(),2))
p = ridge_fit(Xtr,ytr,Xpv,10.0)
print('PSEUDO-VAL 431 ridge MAE:', round(np.abs(p-ypv).mean(),2), 'bias:', round((p-ypv).mean(),2), 'pred std:', round(p.std(),1), 'y std:', round(ypv.std(),1))
for s in [95,207,319,403]:
    m_ = D[D.snapshot_day==s]; Xs = m_[feat].astype(float).fillna(med).values
    ps = ridge_fit(Xtr,ytr,Xs,10.0)
    print(f'  snap {s}: bias {np.round((ps-m_[TARGET]).mean(),2)} MAE {np.round(np.abs(ps-m_[TARGET]).mean(),2)}')
print('corr(pred,y) pv:', round(pd.Series(p).corr(pd.Series(ypv)),3))
Xtr2 = np.hstack([Xtr, ptr[['gbm_pred']].astype(float).values]); Xpv2 = np.hstack([Xpv, pv[['gbm_pred']].astype(float).values])
p2 = ridge_fit(Xtr2,ytr,Xpv2,10.0)
print('pv MAE + gbm_pred:', round(np.abs(p2-ypv).mean(),2))

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
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

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
trd = sorted(snapshot_days()['train'])
feat0 = [c for c in F.columns if c not in key+['gbm_pred']]
def ridge_fit(Xtr, ytr, Xva, alpha=100.0):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym=ytr.mean()
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@(ytr-ym))
    return Zv@w+ym
def wridge_fit(Xtr, ytr, wt, Xva, alpha=100.0):
    mu = (Xtr*wt[:,None]).sum(0)/wt.sum(); sg = np.sqrt(((Xtr-mu)**2*wt[:,None]).sum(0)/wt.sum())+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym=(ytr*wt).sum()/wt.sum()
    ZtW = Z*wt[:,None]
    w = np.linalg.solve(Z.T@ZtW+alpha*np.eye(Z.shape[1]), ZtW.T@(ytr-ym))
    return Zv@w+ym
def build_X(df, cols, med): return df[cols].astype(float).fillna(med).values

# market aggregates per snapshot (from D rows at that snapshot = observable at s)
mkt28 = D.groupby('snapshot_day')['spend_28'].mean()
mkt_prev = mkt28.shift(1)  # snapshot s-28's trailing-28d market mean = window [s-56, s-28]
D['mkt28'] = D['snapshot_day'].map(mkt28)
D['mkt_mom'] = (D['snapshot_day'].map(mkt28)/D['snapshot_day'].map(mkt_prev)).replace([np.inf,-np.inf],np.nan)
D['drift28'] = D['spend_28'].astype(float)/D['mkt28']
D['drift_x'] = D['drift28']*np.log1p(D['spend_28'].astype(float))
mcols = ['mkt28','mkt_mom','drift28','drift_x']
mm = D[mcols].median()
print('mkt_mom by snap:', D.groupby('snapshot_day')['mkt_mom'].first().round(3).to_dict())

def run_variant(add_cols, weighted=None, snaps_test=[375,403,431], alpha=100.0):
    maes=[]
    for s in snaps_test:
        ptr = D[(D.snapshot_day.isin(trd)) & (D.snapshot_day < s)]
        pv = D[D.snapshot_day==s]
        cols = feat0 + [c for c in add_cols if c in D.columns]
        med = ptr[cols].astype(float).median()
        Xtr = build_X(ptr, cols, med); Xpv = build_X(pv, cols, med)
        ytr = ptr[TARGET].values; ypv = pv[TARGET].values
        if weighted:
            wt = 0.5**((s-ptr['snapshot_day'].values)/weighted)
            p = wridge_fit(Xtr,ytr,wt,Xpv,alpha)
        else:
            p = ridge_fit(Xtr,ytr,Xpv,alpha)
        maes.append(np.abs(p-ypv).mean())
    return maes

for name, add, w in [('base',[],None), ('base+w150',[],150), ('drift',['mkt28','mkt_mom','drift28'],None),
                     ('drift+w150',['mkt28','mkt_mom','drift28'],150), ('drift+dx',['mkt28','mkt_mom','drift28','drift_x'],None),
                     ('drift+w150+dx',['mkt28','mkt_mom','drift28','drift_x'],150),
                     ('drift+w150+dx+shrink',['mkt28','mkt_mom','drift28','drift_x'],150)]:
    if 'shrink' in name:
        k=16; gmed = D[D.snapshot_day.isin(trd)][TARGET].median()
        D['shr'] = (D['trips_84'].astype(float)/(D['trips_84'].astype(float)+k))*D['spend_28'].astype(float) + (1-D['trips_84'].astype(float)/(D['trips_84'].astype(float)+k))*gmed
        add = add+['shr']
    m = run_variant(add, w)
    print(f'{name:24s} MAE per snap {[round(x,2) for x in m]} avg {np.mean(m):.3f}')

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days, save_table
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
trd = sorted(snapshot_days()['train'])
feat0 = [c for c in F.columns if c not in key+['gbm_pred']]
def ridge_fit(Xtr, ytr, Xva, alpha=100.0):
    mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sg; Zv=(Xva-mu)/sg
    ym=ytr.mean()
    w = np.linalg.solve(Z.T@Z+alpha*np.eye(Z.shape[1]), Z.T@(ytr-ym))
    return Zv@w+ym
def build_X(df, cols, med): return df[cols].astype(float).fillna(med).values

# market features from saved table (within-snapshot aggregates of observable features)
mkt28 = D.groupby('snapshot_day')['spend_28'].mean()
mkt_prev = mkt28.shift(1)
D['mkt28'] = D['snapshot_day'].map(mkt28)
D['mkt_mom'] = (D['snapshot_day'].map(mkt28)/D['snapshot_day'].map(mkt_prev)).replace([np.inf,-np.inf],np.nan)
D['drift28'] = D['spend_28'].astype(float)/D['mkt28']
gmed = D[D.snapshot_day.isin(trd)][TARGET].median()
k=16; w_ = D['trips_84'].astype(float)/(D['trips_84'].astype(float)+k)
D['shr'] = w_*D['spend_28'].astype(float) + (1-w_)*gmed
newcols = ['mkt28','mkt_mom','drift28','shr']
print(D[newcols].describe().round(3))

def run_variant(add_cols, snaps_test, alpha):
    maes=[]
    for s in snaps_test:
        ptr = D[(D.snapshot_day.isin(trd)) & (D.snapshot_day < s)]; pv = D[D.snapshot_day==s]
        cols = feat0 + add_cols
        med = ptr[cols].astype(float).median()
        p = ridge_fit(build_X(ptr,cols,med), ptr[TARGET].values, build_X(pv,cols,med), alpha)
        maes.append(np.abs(p-pv[TARGET].values).mean())
    return np.mean(maes), [round(x,2) for x in maes]
snaps_test=[347,375,403,431]
for name, add in [('base',[]), ('+mkt4',newcols)]:
    for a in [30,100,300]:
        avg, per = run_variant(add, snaps_test, a)
        print(f'{name:8s} alpha={a:4d}: avg {avg:.3f} per {per}')
# save final candidate table (all 17 snapshot rows)
OUT = F[key+feat0].copy()
for c in newcols: OUT[c] = D[c].values
OUT['mkt_mom'] = OUT['mkt_mom'].fillna(OUT['mkt_mom'].median())
path = save_table(OUT, 'e015_market_ctx')
print('saved:', path, OUT.shape)
print('sanity: rows per snap', OUT.groupby('snapshot_day').size().to_dict())

# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, TARGET, snapshot_days, save_table
tt = train_targets(); key=['household_key','snapshot_day']
F = load_saved('e013_denoise.parquet')
D = F.merge(tt, on=key)
trd = sorted(snapshot_days()['train'])
feat0 = [c for c in F.columns if c not in key+['gbm_pred']]
mkt28 = D.groupby('snapshot_day')['spend_28'].mean()
mkt_prev = mkt28.shift(1)
D['mkt28'] = D['snapshot_day'].map(mkt28)
D['mkt_mom'] = (D['snapshot_day'].map(mkt28)/D['snapshot_day'].map(mkt_prev)).replace([np.inf,-np.inf],np.nan)
D['drift28'] = D['spend_28'].astype(float)/D['mkt28']
gmed = D[D.snapshot_day.isin(trd)][TARGET].median()
k=16; w_ = D['trips_84'].astype(float)/(D['trips_84'].astype(float)+k)
D['shr'] = w_*D['spend_28'].astype(float) + (1-w_)*gmed
newcols = ['mkt28','mkt_mom','drift28','shr']
OUT = F[key+feat0].merge(D[key+newcols], on=key, how='left')
OUT['mkt_mom'] = OUT['mkt_mom'].fillna(OUT['mkt_mom'].median())
assert OUT.shape[0]==36426 and not OUT[key].duplicated().any()
assert sorted(OUT.snapshot_day.unique())==sorted(F.snapshot_day.unique())
path = save_table(OUT, 'e015_market_ctx')
print('saved:', path, OUT.shape, '| nan%:', round(OUT.isna().mean().mean()*100,2))
print('snap rows:', OUT.groupby('snapshot_day').size().to_dict())