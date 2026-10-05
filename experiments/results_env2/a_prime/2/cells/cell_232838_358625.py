import agent_api as A, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')

# ---------- histogram GBM (numpy) ----------
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

def gbm_pred(base, trees, Xb):
    out=np.full(len(Xb), base)
    for nodes in trees:
        o=np.empty(len(Xb))
        def prd(ni, idx):
            nd=nodes[ni]
            if nd[0]=='leaf': o[idx]=nd[1]; return
            _,j,t,l,r=nd; m=Xb[idx,j]<=t
            prd(l,idx[m]); prd(r,idx[~m])
        prd(0, np.arange(len(Xb)))
        out+=0.08*o
    return out

def ridge_pred(Xtr,ytr,Xva,lam=3.0):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-9
    Z=(Xtr-mu)/sd; Zv=(Xva-mu)/sd
    A_=Z.T@Z+lam*np.eye(Z.shape[1]); b=Z.T@ytr
    return Zv@np.linalg.solve(A_,b)

# ---------- data ----------
t11 = A.load_saved('e011_discounts.parquet')
tt = A.train_targets()
m = tt.merge(t11, on=['household_key','snapshot_day'], how='inner')
feats=[c for c in t11.columns if c not in ('household_key','snapshot_day')]
strf=[c for c in feats if m[c].dtype==object]
print('string feats:', strf)
for c in strf: m[c]=m[c].astype('category').cat.codes.replace(-1,np.nan)
X=m[feats].astype(float).values; y=m['future_spend_4w'].values; sd=m['snapshot_day'].values
tr431 = sd<=403; va431 = sd==431
tr403 = sd<=375; va403 = sd==403
print('holdout431: tr',tr431.sum(),'va',va431.sum(),'| holdout403: tr',tr403.sum(),'va',va403.sum())

t0=time.time()
Xb,_=make_bins(X)
base,trees=fit_gbm(Xb[tr431],y[tr431])
p=gbm_pred(base,trees,Xb[va431])
print('GBM holdout431 MAE: %.3f  (%.0fs)'%(np.abs(p-y[va431]).mean(), time.time()-t0))
t0=time.time()
base,trees=fit_gbm(Xb[tr403],y[tr403])
p=gbm_pred(base,trees,Xb[va403])
print('GBM holdout403 MAE: %.3f  (%.0fs)'%(np.abs(p-y[va403]).mean(), time.time()-t0))
# ridge with one-hot snapshot
Oh=np.zeros((len(m),13))
for i,d in enumerate(sorted(m.snapshot_day.unique())): Oh[sd==d,i]=1
Xr=np.hstack([X,Oh])
pr=ridge_pred(Xr[tr431],y[tr431],Xr[va431])
print('Ridge holdout431 MAE: %.3f'%np.abs(pr-y[va431]).mean())
# E009 columns
e9=A.load_saved('e009_ar_season.parquet')
print('e009 cols:', list(e9.columns))