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
sq2=np.clip(sq_oof,0,None)**2; rawp=np.clip(raw_oof,0,None)
# per-snapshot weighted blend: choose w per snapshot via leave-self-out of snapshot residuals
snap_ids = sorted(set(days[is_tr]))
ws = np.arange(0.5,1.01,0.05)
best_w={}
for s in snap_ids:
    msk = is_tr&(days!=s)
    scores=[]
    for w in ws:
        p=w*sq2+(1-w)*rawp
        scores.append(np.mean(np.abs(p[msk]-y[msk])))
    best_w[s]=ws[int(np.argmin(scores))]
print('per-snap best w:', {s:round(v,2) for s,v in best_w.items()})
p_adapt = np.array([best_w.get(s,0.75)*sq2[i]+(1-best_w.get(s,0.75))*rawp[i] for i,s in enumerate(days)])
print('adaptive-w blend OOF MAE %.3f | late %.3f'%(ev(p_adapt),ev(p_adapt,late)))
# also try: residual-based per-snapshot multiplicative level calibration in raw space
lvl={s:y[is_tr&(days==s)].mean() for s in snap_ids}
lvo={s:np.mean([v for s2,v in lvl.items() if s2!=s]) for s in snap_ids}
p_lvl = rawp*np.array([lvo.get(s,1.0) for s in days])/np.array([lvl.get(s,1.0) for s in days])
print('level-calibrated raw MAE %.3f'%ev(p_lvl))
# sqrt-space level calibration
sqc = sq2*np.array([lvo.get(s,1.0) for s in days])/np.array([lvl.get(s,1.0) for s in days])
print('level-calibrated sqrt2 MAE %.3f'%ev(sqc))
# sqrt + small raw: fine grid
for w in [0.7,0.75,0.8]:
    p=w*sq2+(1-w)*rawp
    print('w=%.2f MAE %.3f late %.3f'%(w,ev(p),ev(p,late)))
print('t',round(time.time()-t0,1))
