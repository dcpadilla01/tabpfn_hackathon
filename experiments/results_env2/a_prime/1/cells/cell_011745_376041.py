import agent_api, pandas as pd, numpy as np, time
t0=time.time()
df = agent_api.load_saved('e015_stack.parquet')
print('e012_style shape:', agent_api.load_saved('e012_style.parquet').shape)
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
Xn = d[num_cols].astype(float).values
Xd = pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)
X = np.hstack([Xn, Xd])
days = d['snapshot_day'].values; y=d['future_spend_4w'].values.astype(float)
tr_days = np.array(sorted(set(days[is_tr])))
late = is_tr & (days>=347)
print('y quantiles (50/90/95/99/99.9):', np.nanpercentile(y[is_tr],[50,90,95,99,99.9]).round(1), 'max', np.nanmax(y[is_tr]))

def std_fit(A):
    mu=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0); sd[~np.isfinite(sd)]=1.0; sd[sd==0]=1.0; return mu,sd
def std_apply(A,mu,sd):
    Z=(A-mu)/sd; return np.where(np.isfinite(Z),Z,0.0)

def loso(Xall, alpha, tfun, mode='loso', decay=None, day_feat=False, wins=True):
    out=np.full(len(d),np.nan)
    for s in tr_days:
        q=np.where(days==s)[0]
        f=np.where(is_tr&(days!=s))[0] if mode=='loso' else np.where(is_tr&(days<s))[0]
        if len(f)<50:
            out[q]=np.nanmean(y[is_tr]); continue
        Xf=Xall[f].copy(); Xq=Xall[q].copy()
        if day_feat:
            Xf=np.hstack([Xf,(days[f]/100.0)[:,None]]); Xq=np.hstack([Xq,(days[q]/100.0)[:,None]])
        if wins:
            lo=np.nanpercentile(Xf,0.5,axis=0); hi=np.nanpercentile(Xf,99.5,axis=0)
            lo=np.where(np.isfinite(lo),lo,0); hi=np.where(np.isfinite(hi),hi,1); bad=hi<=lo; hi[bad]=lo[bad]+1
            Xf=np.clip(Xf,lo,hi); Xq=np.clip(Xq,lo,hi)
        w=np.ones(len(f)) if decay is None else np.exp(-(s-days[f])/decay)
        yf=tfun(y[f])
        mu,sd=std_fit(Xf); Xfs=std_apply(Xf,mu,sd)
        ym=np.average(yf,weights=w); r=(yf-ym)*np.sqrt(w); Xw=Xfs*np.sqrt(w)[:,None]
        G=Xw.T@Xw; b=Xw.T@r
        wg=np.linalg.solve(G+alpha*np.eye(G.shape[0]),b)
        out[q]=std_apply(Xq,mu,sd)@wg+ym
    return out

def ev(oof, inv, mask=None):
    pr=np.clip(inv(np.asarray(oof,float)),0,None)
    m = is_tr if mask is None else mask
    return np.mean(np.abs(pr[m]-y[m]))

tfuns = {
 'id':    (lambda v: v,               lambda p: p),
 'sqrt':  (np.sqrt,                   lambda p: p**2),
 'cbrt':  (np.cbrt,                   lambda p: p**3),
 'log':   (np.log1p,                  np.expm1),
 'as20':  (lambda v: np.arcsinh(v/20.0), lambda p: 20.0*np.sinh(p)),
}
print('--- LOSO a=1000, winsorized, all cols ---')
oof_store={}
for name,(t,inv) in tfuns.items():
    o=loso(X,1000,t,'loso')
    oof_store[name]=o
    print('%-6s train MAE %6.3f | late %6.3f'%(name, ev(o,inv), ev(o,inv,late)))
print('t',round(time.time()-t0,1))
