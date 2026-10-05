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
        ym=np.mean(yf_all[f]); r=yf_all[f]-ym; Xw=Xfs
        wg=np.linalg.solve(Xw.T@Xw+alpha*np.eye(Xw.shape[1]),Xw.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(pred, mask=None):
    pr=np.clip(np.asarray(pred,float),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

y_sqrt=np.sqrt(y)
sq_oof=loso(X,1000,y_sqrt)          # sqrt-space ridge
raw_oof=loso(X,1000,y)              # raw-space ridge
log_oof=loso(X,1000,np.log1p(y))    # log-space ridge
print('OOF MAE: sqrt %.3f | raw %.3f | log %.3f'%(ev(np.clip(sq_oof,0,None)**2), ev(np.clip(raw_oof,0,None)), ev(np.clip(np.expm1(log_oof),0,None))))
res = np.where(is_tr, y_sqrt-np.clip(sq_oof,0,None), np.nan)
mr={s:np.nanmean(res[days==s]) for s in tr_days}
bs={s:np.mean([v for s2,v in mr.items() if s2!=s]) for s in tr_days}
sq_cal = np.clip(sq_oof,0,None)**2 + np.array([bs.get(s,0.0) for s in days])
print('sqrt+biascal OOF MAE %.3f'%ev(sq_cal))
best=(1e9,None)
for w in [0,0.25,0.5,0.75,1.0]:
    p=w*(np.clip(sq_oof,0,None)**2)+(1-w)*np.clip(raw_oof,0,None)
    m=ev(p)
    if m<best[0]: best=(m,w)
    print('blend sqrt w=%.2f MAE %.3f'%(w,m))
p_mix=(np.clip(sq_oof,0,None)+np.clip(raw_oof,0,None))**2/4
print('avg-in-sqrt-space blend MAE %.3f'%ev(p_mix))
w=best[1]
p=w*(np.clip(sq_oof,0,None)**2)+(1-w)*np.clip(raw_oof,0,None)
print('winner(w=%.2f) late MAE %.3f | cur stack late OOF %.3f'%(w,ev(p,late), ev(np.clip(d['stack_ridge'].values,0,None)[is_tr],late)))
print('t',round(time.time()-t0,1))
