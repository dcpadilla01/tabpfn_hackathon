
import agent_api as A, pandas as pd, numpy as np
from collections import defaultdict
tt = A.train_targets()
e8 = A.load_saved('e008_level_shape.parquet')
m = tt.merge(e8, on=['household_key','snapshot_day'], how='left')
y = m['future_spend_4w'].values.astype(float)
hk = m['household_key'].values; sd = m['snapshot_day'].values
med = np.median(y)
def mae(p): return float(np.mean(np.abs(np.asarray(p,float)-y)))
ymap = {(h,s):v for h,s,v in zip(tt.household_key,tt.snapshot_day,tt.future_spend_4w)}
ar1 = np.array([ymap.get((h,s-28),np.nan) for h,s in zip(hk,sd)])
ar2 = np.array([ymap.get((h,s-56),np.nan) for h,s in zip(hk,sd)])
ok1=~np.isnan(ar1); ok2=~np.isnan(ar2)
print('AR1 cov',round(ok1.mean(),3),'MAE fill-med',round(mae(np.where(ok1,ar1,med)),2))
print('AR1+2 blend', round(mae(np.where(ok1&ok2,0.5*ar1+0.5*ar2,np.where(ok1,ar1,med))),2))
byh=defaultdict(list)
for h,s,v in zip(tt.household_key,tt.snapshot_day,tt.future_spend_4w): byh[h].append((s,v))
tsmean=np.full(len(y),np.nan)
for i,(h,s) in enumerate(zip(hk,sd)):
    past=[v for s2,v in byh[h] if s2<s]
    if past: tsmean[i]=np.mean(past)
okt=~np.isnan(tsmean)
print('TSmean cov',round(okt.mean(),3),'MAE',round(mae(np.where(okt,tsmean,med)),2))
print('blend .5sp28+.5tsmean',round(mae(0.5*m['sp28'].fillna(0)+0.5*np.where(okt,tsmean,med)),2))
print('blend .6/.4',round(mae(0.6*m['sp28'].fillna(0)+0.4*np.where(okt,tsmean,med)),2))
print('oracle hh-mean insample',round(mae(m.groupby('household_key')['future_spend_4w'].transform('mean')),2))
print('sp28 vs ar1 corr',round(np.corrcoef(m['sp28'].fillna(0)[ok1],ar1[ok1])[0,1],4),'maxdiff',round(np.abs(m['sp28'].fillna(0)[ok1]-ar1[ok1]).max(),2))
# ridge proxy CV: fit sd<=347, eval 375/403/431
feats=[c for c in e8.columns if c not in('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(e8[c])]
X=m[feats].astype(float); X=X.fillna(X.median()).values
mu=X.mean(0); sg=X.std(0)+1e-9; Xz=(X-mu)/sg
def ridge(lam,tr,ev):
    Xt=Xz[tr]; yt=y[tr]; ym=yt.mean()
    w=np.linalg.solve(Xt.T@Xt+lam*np.eye(len(feats)), Xt.T@(yt-ym))
    return float(np.mean(np.abs(Xz[ev]@w+ym-y[ev])))
tr=sd<=347; ev=(sd>=375)
for lam in [0.3,1,3,10,30,100]:
    print('ridge lam',lam,'MAE 375-431',round(ridge(lam,tr,ev),2))
# splag1y inspection
for c in ['splag1y','sp_lag1y']:
    s=m[c]; ok=s.notna()
    print(c,'cov',round(ok.mean(),3),'corr',round(np.corrcoef(s[ok],y[ok])[0,1],3),'mean',round(s[ok].mean(),1))
print('splag1y cov by snap:',m.groupby('snapshot_day')['splag1y'].apply(lambda x:round(x.notna().mean(),2)).to_string())
# e009 leakage check
e9=A.load_saved('e009_target_enc.parquet')
extra=[c for c in e9.columns if c not in e8.columns]
print('e009 extra cols:',extra)
m9=tt.merge(e9,on=['household_key','snapshot_day'],how='left')
for c in extra:
    s=m9[c].astype(float); ok=s.notna()
    print(c,'cov',round(ok.mean(),3),'corrY',round(np.corrcoef(s[ok],m9['future_spend_4w'].values[ok])[0,1],3))
