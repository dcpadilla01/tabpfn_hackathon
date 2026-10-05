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
invsq = lambda p: p**2

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
        ym=np.mean(yf_all[f]); r=(yf_all[f]-ym); Xw=Xfs
        wg=np.linalg.solve(Xw.T@Xw+alpha*np.eye(Xw.shape[1]),Xw.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(oof, mask=None):
    pr=np.clip(invsq(np.asarray(oof,float)),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

y_sqrt=np.sqrt(y)
print('--- group-by-day ridge, sqrt target, alpha sweep ---')
for a in [100,300,1000,3000]:
    o=loso(X,a,y_sqrt,wins=False)
    print('a=%-5d MAE %.3f late %.3f'%(a,ev(o),ev(o,late)))
groups = {
 'rfm': [c for c in num_cols if any(k in c for k in ['sp','trips','recen','tenur','basket','ew','wk'])],
 'dem': [c for c in num_cols if any(c.startswith(p) for p in ['class','homeowner','kid','has_demo'])],
 'mix': [c for c in num_cols if c.startswith('mix') or 'share' in c],
 'mkt': [c for c in num_cols if any(k in c for k in ['camp','coupon','mailer','display'])],
 'seas':[c for c in num_cols if any(k in c for k in ['block','lag','s336','s364','s392','alltime','avg28'])],
 'dyn': [c for c in num_cols if any(k in c for k in ['d_','mom','ratio','slope','streak','vol'])],
 'style':[c for c in num_cols if any(k in c for k in ['unit','disc','store','hhi','nstore','trip_time','items','nprod'])],
 'knn': [c for c in num_cols if 'knn' in c],
 'stack':[c for c in num_cols if 'stack' in c],
}
print({k:len(v) for k,v in groups.items()})
Xg = {k: d[v].astype(float).values for k,v in groups.items()}
Xg['dum']=pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)
print('--- solo groups ---')
for k,Xs in Xg.items():
    o=loso(Xs,300,y_sqrt,wins=False)
    print('%-6s n=%-4d solo MAE %.3f'%(k,Xs.shape[1],ev(o)))
print('--- cumulative groups ---')
Xc=None
for k in ['rfm','dyn','style','seas','mix','mkt','dem','knn','stack','dum']:
    Xc = Xg[k].copy() if Xc is None else np.hstack([Xc,Xg[k]])
    o=loso(Xc,300,y_sqrt,wins=False)
    print('cum %-6s n=%-4d MAE %.3f'%(k,Xc.shape[1],ev(o)))
print('t',round(time.time()-t0,1))
