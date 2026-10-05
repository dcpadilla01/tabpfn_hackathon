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
tr_days = sorted(set(days[is_tr])); val_days = sorted(set(days[~is_tr]))
print('train days',tr_days,'| val days',val_days)

def fit_pred(Xall,f,q,yf_all,alpha):
    Xf=Xall[f].copy(); Xq=Xall[q].copy()
    lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
    lo=np.where(np.isfinite(lo),lo,0.0); hi=np.where(np.isfinite(hi),hi,1.0)
    bad=hi<=lo; hi[bad]=lo[bad]+1.0
    Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
    mu=np.nanmean(Xf,axis=0); sd=np.nanstd(Xf,axis=0)
    sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0
    Zf=(Xf-mu)/sd; Zf=np.where(np.isfinite(Zf),Zf,0.0)
    Zq=(Xq-mu)/sd; Zq=np.where(np.isfinite(Zq),Zq,0.0)
    yf=yf_all[f]; ym=np.mean(yf)
    wg=np.linalg.solve(Zf.T@Zf+alpha*np.eye(Zf.shape[1]),Zf.T@(yf-ym))
    return Zq@wg+ym

def stack_preds(yf_all, alpha=300.0):
    out=np.full(len(d),np.nan)
    for s in tr_days:   # leave-one-snapshot-out on train rows
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        out[q]=fit_pred(X,f,q,yf_all,alpha)
    for s in val_days:  # validation rows: fit on ALL train rows
        q=np.where(days==s)[0]; f=np.where(is_tr)[0]
        out[q]=fit_pred(X,f,q,yf_all,alpha)
    return out

sq_oof  = stack_preds(np.sqrt(y))
raw_oof = stack_preds(y)
stack_sqrt  = sq_oof
stack_sqrt2 = np.clip(sq_oof,0,None)**2
stack_raw   = np.clip(raw_oof,0,None)
stack_blend = 0.75*stack_sqrt2 + 0.25*stack_raw
def ev(pred, mask=None):
    pr=np.clip(np.asarray(pred,float),0,None); m=is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))
late=is_tr&(days>=347)
print('OOF train MAE: sqrt2 %.3f | raw %.3f | blend %.3f | old log-stack %.3f'%(
    ev(stack_sqrt2),ev(stack_raw),ev(stack_blend),ev(np.clip(d['stack_ridge'].values,0,None))))
print('late MAE: sqrt2 %.3f | blend %.3f'%(ev(stack_sqrt2,late),ev(stack_blend,late)))
print('NaN check:', np.isnan(stack_blend).sum(), np.isnan(stack_sqrt2).sum(), np.isnan(stack_raw).sum())
out = d.drop(columns=['future_spend_4w']).copy()
out['stack_sqrt']=stack_sqrt; out['stack_sqrt2']=stack_sqrt2
out['stack_raw']=stack_raw;   out['stack_blend']=stack_blend
p = agent_api.save_table(out,'e020_sqrtstack.parquet')
print('saved:',p,'shape',out.shape,'t',round(time.time()-t0,1))
