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

def loso(Xall, alpha, tfun, decay=None, day_feat=False, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]
        f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(y[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if day_feat:
            Xf=np.hstack([Xf,(days[f]/100.0)[:,None]]); Xq=np.hstack([Xq,(days[q]/100.0)[:,None]])
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        w=np.ones(len(f)) if decay is None else np.exp(-(s-days[f])/decay)
        yf=tfun(y[f]); mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.average(yf,weights=w); r=(yf-ym)*np.sqrt(w); Xw=Xfs*np.sqrt(w)[:,None]
        wg=np.linalg.solve(Xw.T@Xw+alpha*np.eye(Xw.shape[1]),Xw.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
def ev(oof, inv, mask=None):
    pr=np.clip(inv(np.asarray(oof,float)),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

sq=lambda v:np.sqrt(v); invsq=lambda p:p**2
print('--- sqrt alpha sweep ---')
for a in [100,300,1000,3000,10000]:
    o=loso(X,a,sq); print('a=%-5d MAE %.3f late %.3f'%(a,ev(o,invsq),ev(o,invsq,late)))
print('--- sqrt, no winsorize ---')
o=loso(X,1000,sq,wins=False); print('MAE %.3f late %.3f'%(ev(o,invsq),ev(o,invsq,late)))
print('--- sqrt + day_feat ---')
o=loso(X,1000,sq,day_feat=True); print('MAE %.3f late %.3f'%(ev(o,invsq),ev(o,invsq,late)))
print('--- 4th root ---')
o=loso(X,1000,lambda v:v**0.25); print('MAE %.3f'%(ev(o,lambda p:p**4),))
print('--- two-part: P(zero) logit-ish ridge x sqrt spend given nonzero ---')
z=(y==0).astype(float)
pzero=loso(X,1000,lambda v:z[v if False else slice(None)][f] if False else None) if False else None
# simpler: reuse loso with custom target arrays
def loso_yt(Xall, alpha, ytfun, decay=None, day_feat=False, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]; f=np.where(is_tr&(days!=s))[0]
        if len(f)<50: out[q]=np.nanmean(y[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if day_feat:
            Xf=np.hstack([Xf,(days[f]/100.0)[:,None]]); Xq=np.hstack([Xq,(days[q]/100.0)[:,None]])
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        w=np.ones(len(f)) if decay is None else np.exp(-(s-days[f])/decay)
        yf=ytfun(f); mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.average(yf,weights=w); r=(yf-ym)*np.sqrt(w); Xw=Xfs*np.sqrt(w)[:,None]
        wg=np.linalg.solve(Xw.T@Xw+alpha*np.eye(Xw.shape[1]),Xw.T@r)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out
p_pos = loso_yt(X,1000,lambda f: np.sqrt(y[f]))
p_zero= loso_yt(X,1000,lambda f: z[f])
for th in [0.5,0.6,0.7]:
    pred=np.where(p_zero<th,p_pos,0.0)
    print('two-part th=%.1f MAE %.3f late %.3f'%(th,ev(pred,lambda p:p),ev(pred,lambda p:p,late)))
o=loso(X,1000,sq)
for wgt in [0.0,0.25,0.5,0.75,1.0]:
    pred=wgt*np.where(p_zero<0.6,p_pos,0.0)+(1-wgt)*np.clip(invsq(o),0,None)
    print('blend two-part(%.2f)+sqrt %.2f MAE %.3f'%(wgt,1-wgt,ev(pred,lambda p:p)))
print('t',round(time.time()-t0,1))
