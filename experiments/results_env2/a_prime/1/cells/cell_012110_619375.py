import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = np.array(sorted(set(days[is_tr])))
late = is_tr & (days>=347)

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd; return np.where(np.isfinite(Z),Z,0.0)

def loso(Xall, alpha, yf_all, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(yf_all[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.mean(yf_all[f]); r=yf_all[f]-ym
        wg=np.linalg.solve(Xfs.T@Xfs+alpha*np.eye(Xfs.shape[1]),Xfs.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(pred, mask=None):
    pr=np.clip(np.asarray(pred,float),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

y_sqrt=np.sqrt(y)
sq_oof=loso(X,1000,y_sqrt)
raw_oof=loso(X,1000,y)
log_oof=loso(X,1000,np.log1p(y))
sq2=np.clip(sq_oof,0,None)**2
rawp=np.clip(raw_oof,0,None)
logp=np.clip(np.expm1(log_oof),0,None)
print('members: sqrt2 %.3f | raw %.3f | logexp %.3f | old stack_ridge %.3f'%(
    ev(sq2),ev(rawp),ev(logp),ev(np.clip(d['stack_ridge'].values,0,None))))
best=(1e9,None)
for w1 in [0.5,0.6,0.7,0.8,0.9,1.0]:
    for w2 in [0,0.1,0.2,0.3,0.4]:
        w3=1-w1-w2
        if w3<-1e-9 or w3>0.3: continue
        p=w1*sq2+w2*rawp+w3*logp
        m=ev(p)
        if m<best[0]: best=(m,(w1,w2,w3))
print('best 3-blend', best)
# fine 2-blend
best2=(1e9,None)
for w in np.arange(0.5,1.01,0.05):
    p=w*sq2+(1-w)*rawp; m=ev(p)
    if m<best2[0]: best2=(m,round(w,2))
print('best 2-blend sqrt2+raw', best2)
w=best2[1]
p=w*sq2+(1-w)*rawp
print('2-blend late MAE %.3f | per-day means:'%ev(p,late))
print(pd.DataFrame({'day':days[is_tr],'y':y[is_tr],'p':p[is_tr]}).groupby('day').mean().round(1).T)
print('NaNs in sq2/rawp:', np.isnan(sq2).sum(), np.isnan(rawp).sum())
print('t',round(time.time()-t0,1))
