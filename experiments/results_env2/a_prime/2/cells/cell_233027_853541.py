import agent_api as A, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
def make_bins(X, n_bins=32):
    n,p = X.shape; NANB = n_bins
    Xb = np.empty((n,p), dtype=np.int16); edges=[]
    for j in range(p):
        x = X[:,j]; ok = ~np.isnan(x)
        if ok.sum()<50: Xb[:,j]=NANB; edges.append(None); continue
        e = np.unique(np.quantile(x[ok], np.linspace(0,1,n_bins+1)[1:-1]))
        col = np.full(n, NANB, dtype=np.int16); col[ok] = np.searchsorted(e, x[ok], side='right')
        Xb[:,j]=col; edges.append(e)
    return Xb, edges
def fit_gbm(Xb, y, n_trees=200, lr=0.08, depth=4, min_leaf=20, seed=0, cols_frac=0.8):
    rng = np.random.RandomState(seed); n, p = Xb.shape
    base = y.mean(); pred = np.full(n, base); trees=[]; NB=33; gains={}
    for it in range(n_trees):
        resid = y - pred
        ridx = rng.choice(n, int(0.8*n), replace=False)
        cols = rng.choice(p, max(5,int(cols_frac*p)), replace=False)
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
            gains[j]=gains.get(j,0)+best[2]
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
    return base, trees, gains
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

t11 = A.load_saved('e011_discounts.parquet'); tt = A.train_targets()
m = tt.merge(t11, on=['household_key','snapshot_day'], how='inner')
feats=[c for c in t11.columns if c not in ('household_key','snapshot_day')]
for c in feats:
    if m[c].dtype==object: m[c]=m[c].astype('category').cat.codes.replace(-1,np.nan)
X=m[feats].astype(float).values; y=m['future_spend_4w'].values; sd=m['snapshot_day'].values
print("index col sample:", m['index'].values[:5], m['index'].values[-5:])
tr431 = sd<=403; va431 = sd==431
Xb,_=make_bins(X)
base,trees,gains=fit_gbm(Xb[tr431],y[tr431])
p=gbm_pred(base,trees,Xb[va431]); yv=y[va431]
print('holdout431 MAE %.3f'%np.abs(p-yv).mean())
imp=sorted(gains.items(), key=lambda kv:-kv[1])[:20]
print('top gain feats:', [(feats[j],int(g)) for j,g in imp])
dsl=m['days_since_last'].values[va431]
for lo,hi in [(0,7),(7,14),(14,30),(30,60),(60,200)]:
    mk=(dsl>=lo)&(dsl<hi)
    if mk.sum(): print('dsl[%3d,%3d) n=%4d MAE=%6.1f  pred_mean=%6.1f y_mean=%6.1f zero%%=%.2f'%(lo,hi,mk.sum(),np.abs(p[mk]-yv[mk]).mean(),p[mk].mean(),yv[mk].mean(),(yv[mk]==0).mean()))
for lo,hi in [(0,1),(1,50),(50,150),(150,400),(400,2500)]:
    mk=(yv>=lo)&(yv<hi)
    print('y[%4d,%5d) n=%4d MAE=%6.1f pred_mean=%6.1f'%(lo,hi,mk.sum(),np.abs(p[mk]-yv[mk]).mean(),p[mk].mean()))
keep=[j for j,f in enumerate(feats) if f!='index']
base,trees,_=fit_gbm(Xb[:,keep][tr431],y[tr431]); p2=gbm_pred(base,trees,Xb[:,keep][va431])
print('no-index MAE %.3f'%np.abs(p2-yv).mean())
yl=np.log1p(y)
base,trees,_=fit_gbm(Xb[tr431],yl[tr431]); pl=np.expm1(gbm_pred(base,trees,Xb[va431]))
print('log-target MAE %.3f'%np.abs(pl-yv).mean())