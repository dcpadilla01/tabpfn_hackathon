
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
F = agent_api.load_saved('e008_candidate.parquet')
tt = agent_api.train_targets()
df = F.merge(tt, on=['household_key','snapshot_day'])
itr = (df.snapshot_day<=347).values; iva = (df.snapshot_day>=375).values
y = df['future_spend_4w'].values
E5COLS = ['spend_7','spend_14','spend_28','spend_56','spend_84','spend_180','spend_365','spend_28_prior','spend_84_prior',
'baskets_28','baskets_84','days_since_last','days_since_first','avg_basket_84','trips_per_wk_84','spend_28_ratio',
'n_products_84','n_stores_84','spend_trend','active_28','ew_7','ew_14','ew_28','ew_56','ew_84','ew_180',
'spend_lag336','spend_lag364','spend_lag392','longrun_wk','ratio28_lr','ratio84_lr','basket_max_84','basket_std_84',
'basket_med_84','active_days_28','gap_cv']
def ridge(Xtr,ytr,Xva,lam,clip=8.0):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-8]=1
    A=np.clip((Xtr-mu)/sd,-clip,clip); B=np.clip((Xva-mu)/sd,-clip,clip)
    A=np.c_[np.ones(len(A)),A]; B=np.c_[np.ones(len(B)),B]
    return B@np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@ytr)
def ev(cols):
    X = df[cols].replace([np.inf,-np.inf],np.nan)
    X = X.fillna(X[itr].median()).fillna(0.0).values.astype(float)
    p = ridge(X[itr],y[itr],X[iva],100.0)
    return np.abs(p-y[iva]).mean()
base = E5COLS
C = ['avg_basket_28','basket_28_vs_84','lines_per_basket_84','units_per_basket_84','unit_price_84']
S = ['sin_y','cos_y','sin_q','cos_q','day_idx']
Sy = ['sin_y','cos_y','day_idx']
df['sin_y2']=np.sin(4*np.pi*df['snapshot_day']/364); df['cos_y2']=np.cos(4*np.pi*df['snapshot_day']/364)
df['ix1']=df['ew_28']*df['sin_y']; df['ix2']=df['ew_28']*df['cos_y']
df['ix3']=df['spend_84']*df['sin_y']; df['ix4']=df['spend_84']*df['cos_y']
print('base+C+S      %.3f' % ev(base+C+S))
print('base+C+S+har2 %.3f' % ev(base+C+S+['sin_y2','cos_y2']))
print('base+C+S+ix12 %.3f' % ev(base+C+S+['ix1','ix2']))
print('base+C+S+ix34 %.3f' % ev(base+C+S+['ix3','ix4']))
print('base+C+Sy     %.3f' % ev(base+C+Sy))
print('base+Sy       %.3f' % ev(base+Sy))
