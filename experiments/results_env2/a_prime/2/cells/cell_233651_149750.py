import agent_api as A, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
def make_bins(X, n_bins=32):
    n,p = X.shape; NANB = n_bins
    Xb = np.empty((n,p), dtype=np.int16)
    for j in range(p):
        x = X[:,j]; ok = ~np.isnan(x)
        if ok.sum()<50: Xb[:,j]=NANB; continue
        e = np.unique(np.quantile(x[ok], np.linspace(0,1,n_bins+1)[1:-1]))
        col = np.full(n, NANB, dtype=np.int16); col[ok] = np.searchsorted(e, x[ok], side='right')
        Xb[:,j]=col
    return Xb
def fit_gbm(Xb, y, n_trees=200, lr=0.08, depth=4, min_leaf=20, seed=0):
    rng = np.random.RandomState(seed); n, p = Xb.shape
    base = y.mean(); pred = np.full(n, base); trees=[]; NB=33
    for it in range(n_trees):
        resid = y - pred
        ridx = rng.choice(n, int(0.8*n), replace=False)
        cols = rng.choice(p, max(5,int(0.8*p)), replace=False)
        nodes=[]
        def rec(idx, dep):
            i=len(nodes); nodes.append(['leaf', resid[idx].mean()])
            if dep>=depth or len(idx)<2*min_leaf: return i
            best=(None,None,-1e18)
            for j in cols:
                b=Xb[idx,j]
                hsum=np.bincount(b,weights=resid[idx],minlength=NB).astype(float)
                hcnt=np.bincount(b,minlength=NB).astype(float)
                cs=np.cumsum(hsum); cc=np.cumsum(hcnt)
                cl=cc[:-1]; sl=cs[:-1]; cr=cc[-1]-cl; sr=cs[-1]-sl
                gain=np.where((cl>=min_leaf)&(cr>=min_leaf), sl*sl/(cl+1e-9)+sr*sr/(cr+1e-9), -1e18)
                t=int(np.argmax(gain))
                if gain[t]>best[2]: best=(j,t,gain[t])
            j,t,_=best
            if j is None: return i
            mask=Xb[idx,j]<=t; li,ri=idx[mask],idx[~mask]
            if len(li)<min_leaf or len(ri)<min_leaf: return i
            nodes[i]=['split',j,t,None,None]
            nodes[i][3]=rec(li,dep+1); nodes[i][4]=rec(ri,dep+1)
            return i
        rec(ridx,0)
        out=np.empty(n)
        def prd(ni, idx):
            nd=nodes[ni]
            if nd[0]=='leaf': out[idx]=nd[1]; return
            _,j,t,l,r=nd; m=Xb[idx,j]<=t
            prd(l,idx[m]); prd(r,idx[~m])
        prd(0, np.arange(n))
        trees.append(nodes); pred += lr*out
    return base, trees
def gbm_pred(base, trees, Xb, lr=0.08):
    out=np.full(len(Xb), base)
    for nodes in trees:
        o=np.empty(len(Xb))
        def prd(ni, idx):
            nd=nodes[ni]
            if nd[0]=='leaf': o[idx]=nd[1]; return
            _,j,t,l,r=nd; m=Xb[idx,j]<=t
            prd(l,idx[m]); prd(r,idx[~m])
        prd(0, np.arange(len(Xb)))
        out+=lr*o
    return out
def irls_mae(Z, y, iters=15):
    w = np.linalg.lstsq(Z, y, rcond=None)[0]
    for _ in range(iters):
        r = y - Z@w; a = np.maximum(np.abs(r), 1.0); W = 1.0/a
        w = np.linalg.solve((Z*W[:,None]).T@Z + 1e-6*np.eye(Z.shape[1]), (Z*W[:,None]).T@y)
    return w

t11 = A.load_saved('e011_discounts.parquet'); tt = A.train_targets()
m = tt.merge(t11, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats=[c for c in t11.columns if c not in ('household_key','snapshot_day')]
for c in feats:
    if m[c].dtype==object: m[c]=m[c].astype('category').cat.codes.replace(-1,np.nan)
y=m['future_spend_4w'].values; sd=m['snapshot_day'].values; sdays=sorted(m.snapshot_day.unique())
tr431=sd<=403; va431=sd==431

# ---- causal stack v2: richer SF, fallback for s=95 ----
SF=['ew_28','ew_56','ew_84','lr_mean28','lr_med28','spend_28_prior','spend_84_prior','active_28','days_since_last','baskets_28','baskets_84','avg_basket_84','n_products_84','win_cv','spend_lag364','spend_lag336']
Zdf=m[SF].astype(float).copy()
for c in SF:
    Zdf[c]=Zdf[c].fillna(0.0)
Zdf['days_since_last']=Zdf['days_since_last'].clip(0,150)
Z=Zdf.values
def design(Zm):
    return np.hstack([np.ones((len(Zm),1)), Zm, (Zm[:,0]/100.0)**2[:,None], (Zm[:,0]*Zm[:,3]/1e4)[:,None]])
stack=np.full(len(m), np.nan); meta=[]
t0=time.time()
for i,s in enumerate(sdays):
    fit_mask = sd <= s-28; pred_mask = sd == s
    if fit_mask.sum()>500:
        w=irls_mae(design(Z[fit_mask]), y[fit_mask])
        stack[pred_mask]=np.maximum(design(Z[pred_mask])@w, 0)
        meta.append((s, fit_mask.sum(), np.abs(stack[pred_mask]-y[pred_mask]).mean()))
    else:
        stack[pred_mask]=Z[pred_mask,0]  # fallback: ew_28
        meta.append((s, 0, np.abs(stack[pred_mask]-y[pred_mask]).mean()))
print('stack built %.1fs'%(time.time()-t0))
print('per-snapshot stack MAE:', [(s,n,round(e,1)) for s,n,e in meta])
ok=~np.isnan(stack)
print('stack corr %.3f | stack-only MAE(all-train) %.2f'%(np.corrcoef(stack[ok],y[ok])[0,1], np.abs(stack[ok]-y[ok]).mean()))
print('stack-only holdout431 MAE: %.3f'%np.abs(stack[va431]-y[va431]).mean())

X=m[feats].astype(float).values
def run_both(cols_extra, name):
    Xe=np.hstack([X]+[c[:,None] for c in cols_extra]) if cols_extra else X
    Xb=make_bins(Xe)
    b,tr=fit_gbm(Xb[tr431],y[tr431]); p=gbm_pred(b,tr,Xb[va431]); m1=np.abs(p-y[va431]).mean()
    # ridge standardized
    Oh=np.zeros((len(m),12)); u=sorted(set(sd[tr431]))
    for i,d in enumerate(u): Oh[tr431,i]= (sd[tr431]==d); Oh[va431,i]=(sd[va431]==d)
    Xr=np.hstack([Xe,Oh]); mu=Xr[tr431].mean(0); s2=Xr[tr431].std(0)+1e-9
    Zt=(Xr[tr431]-mu)/s2; Zv=(Xr[va431]-mu)/s2
    for lam in [3.0]:
        w=np.linalg.solve(Zt.T@Zt+lam*np.eye(Zt.shape[1]), Zt.T@y[tr431])
        m2=np.abs(Zv@w-y[va431]).mean()
    print('%-28s GBM %.3f | ridge %.3f'%(name,m1,m2))
    return m1,m2
run_both([], 'E011 as-is')
run_both([stack], 'E011+stack')
run_both([np.log1p(stack)], 'E011+log1p(stack)')