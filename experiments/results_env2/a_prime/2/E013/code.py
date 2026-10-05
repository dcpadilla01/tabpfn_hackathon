import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width', 250)
t = A.load_saved('e011_discounts.parquet')
tt = A.train_targets()
print('E011', t.shape, '| targets', tt.shape)
cols = list(t.columns)
print('COLS:', cols)
m = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w'].astype(float).values
print('merged', m.shape, '| zero-frac %.3f' % (y==0).mean())
print(pd.Series(y).describe(percentiles=[.25,.5,.75,.9,.95,.99]).round(1))
print(m.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median']).round(1).T)
num_cols = [c for c in cols if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
str_cols = [c for c in cols if c not in ('household_key','snapshot_day') and c not in num_cols]
print('num', len(num_cols), '| str', str_cols)
rows=[]
for c in num_cols:
    v = m[c].astype(float); vn = v.fillna(v.median()).values
    if vn.std()==0: continue
    rows.append((c, float(np.corrcoef(vn,y)[0,1]), float(np.abs(vn-y).mean())))
d = pd.DataFrame(rows, columns=['feat','corr','naiveMAE'])
d = d.reindex(d.corr.abs().sort_values(ascending=False).index)
print(d.head(28).round(3).to_string())

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
t = A.load_saved('e011_discounts.parquet')
tt = A.train_targets()
m = tt.merge(t, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w'].astype(float).values
num_cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(m[c])]
rows=[]
for c in num_cols:
    v = m[c].astype(float); vn = v.fillna(v.median()).values
    if vn.std()==0: continue
    rows.append((c, float(np.corrcoef(vn,y)[0,1]), float(np.abs(vn-y).mean())))
d = pd.DataFrame(rows, columns=['feat','corr','naiveMAE']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(d.round(3).to_string())
# zero vs nonzero structure
z = y==0
print('\nzero rows:', z.sum())
for c in ['spend_28','spend_84','ew_28','ew_84','days_since_last','active_28','spend_56']:
    v = m[c].astype(float).fillna(m[c].median()).values
    print(f'{c:16s} mean|z={v[z].mean():8.2f} |nz={v[~z].mean():8.2f} | corr_nonz={np.corrcoef(v[~z],y[~z])[0,1]:.3f}')

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
t12 = A.load_saved('e012_hh_target_enc.parquet')
t11 = A.load_saved('e011_discounts.parquet')
tt = A.train_targets()
print('t12', t12.shape, 't11', t11.shape)
new = [c for c in t12.columns if c not in t11.columns]
print('new cols:', new)
m = t12.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>431]
print('train rows', len(tr), 'val rows', len(va))
for c in new:
    print('\n--', c)
    print(' train: nan%%=%.1f mean=%.2f std=%.2f' % (tr[c].isna().mean()*100, tr[c].mean(), tr[c].std()))
    print(' val  : nan%%=%.1f mean=%.2f std=%.2f' % (va[c].isna().mean()*100, va[c].mean(), va[c].std()))
    v = tr[c].values; y = tr['future_spend_4w'].values
    ok = ~np.isnan(v)
    print(' train corr with y: %.3f' % np.corrcoef(v[ok], y[ok])[0,1])
# check hh_mean_loo construction on a sample household
hh = tr.household_key.iloc[0]
sub = m[m.household_key==hh][['snapshot_day','future_spend_4w','hh_mean_loo','hh_n','hh_mean_causal','hh_mean_last3']]
print('\nsample household rows:\n', sub.to_string())
# alignment check: does t12 have same rows as t11?
k11 = set(map(tuple, t11[['household_key','snapshot_day']].values))
k12 = set(map(tuple, t12[['household_key','snapshot_day']].values))
print('\nrows equal:', k11==k12, '| only in 12:', len(k12-k11), '| only in 11:', len(k11-k12))

# ---- cell ----
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

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
t11 = A.load_saved('e011_discounts.parquet')
tt = A.train_targets()
m = tt.merge(t11, on=['household_key','snapshot_day'], how='inner')
feats=[c for c in t11.columns if c not in ('household_key','snapshot_day')]
for c in feats:
    if m[c].dtype==object: m[c]=m[c].astype('category').cat.codes.replace(-1,np.nan)
X=m[feats].astype(float)
inf_cols=[c for c in feats if np.isinf(X[c].values).any()]
print('inf cols:', inf_cols)
X=X.replace([np.inf,-np.inf], np.nan)
# nan counts
nac=X.isna().mean().sort_values(ascending=False)
print('top nan:', dict(nac.head(6).round(3)))
Xv=X.values; y=m['future_spend_4w'].values; sd=m['snapshot_day'].values

# ---- aggregate weekly seasonality (capped view 459) ----
snap=A.snapshot()
tx=snap.table('transactions')
wk=(tx['day']+8)//7
g=tx.groupby(wk)['sales_value'].sum()
idx=np.arange(1, g.index.max()+1)
ser=g.reindex(idx).fillna(0).values
print('\nweeks:', len(ser))
# mean spend by week mod 52
ph=np.arange(len(ser))%52
prof=pd.Series(ser).groupby(ph).mean()
print('seasonal profile (week-mod-52): min=%.0f max=%.0f mean=%.0f, peak weeks:'%(prof.min(),prof.max(),prof.mean()), list(prof.nlargest(6).index), 'trough:', list(prof.nsmallest(4).index))
# lag-52 autocorr of weekly totals
s=ser-ser.mean(); ac52=np.corrcoef(s[:-52],s[52:])[0,1]; ac1=np.corrcoef(s[:-1],s[1:])[0,1]
print('autocorr lag52=%.3f lag1=%.3f'%(ac52,ac1))
# trend: yearly totals
yr=np.arange(len(ser))//52
print('total spend yr1=%.1fM yr2=%.1fM'%(pd.Series(ser).groupby(yr).sum().values[0]/1e6, pd.Series(ser).groupby(yr).sum().values[1] if len(pd.Series(ser).groupby(yr).sum())>1 else -1))

# ---- cell ----
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

# ---- cell ----
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
def irls_mae(Z, y, iters=12):
    w = np.ones(Z.shape[1])
    for _ in range(iters):
        r = y - Z@w
        a = np.maximum(np.abs(r), 1.0)
        W = 1.0/a
        ZtWZ = (Z*W[:,None]).T@Z + 1e-6*np.eye(Z.shape[1])
        w = np.linalg.solve(ZtWZ, (Z*W[:,None]).T@y)
    return w

t11 = A.load_saved('e011_discounts.parquet'); tt = A.train_targets()
m = tt.merge(t11, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats=[c for c in t11.columns if c not in ('household_key','snapshot_day')]
for c in feats:
    if m[c].dtype==object: m[c]=m[c].astype('category').cat.codes.replace(-1,np.nan)
y=m['future_spend_4w'].values; sd=m['snapshot_day'].values
sdays=sorted(m.snapshot_day.unique())
# ---- causal stack_pred: at snapshot s, fit on rows with snapshot s' <= s-28, predict rows at s ----
stack=np.full(len(m), np.nan)
SF=['ew_28','ew_56','ew_84','lr_mean28','spend_28_prior','active_28','days_since_last']
Zall=m[SF].astype(float).copy()
Zall['days_since_last']=Zall['days_since_last'].fillna(120).clip(0,120)
t0=time.time()
for s in sdays:
    fit_mask = sd <= s-28
    pred_mask = sd == s
    Zf=Zall[fit_mask].values; yf=y[fit_mask]
    Zf=np.hstack([np.ones((len(Zf),1)), Zf, Zf[:,0:1]**2/1000.0])
    w=irls_mae(Zf,yf)
    Zp=Zall[pred_mask].values
    Zp=np.hstack([np.ones((len(Zp),1)), Zp, Zp[:,0:1]**2/1000.0])
    stack[pred_mask]=np.maximum(Zp@w, 0)
print('stack built %.0fs'%(time.time()-t0))
ok=~np.isnan(stack)
print('stack corr with y: %.3f | stack-only MAE(train): %.2f'%(np.corrcoef(stack[ok],y[ok])[0,1], np.abs(stack[ok]-y[ok]).mean()))
tr431=sd<=403; va431=sd==431
X=m[feats].astype(float).values
def proxy(cols_extra, name):
    Xe=np.hstack([X]+[c[:,None] for c in cols_extra])
    Xb=make_bins(Xe)
    b,tr=fit_gbm(Xb[tr431],y[tr431]); p=gbm_pred(b,tr,Xb[va431])
    print('%s holdout431 MAE %.3f'%(name,np.abs(p-y[va431]).mean()))
    return np.abs(p-y[va431]).mean()
proxy([], 'E011 baseline')
proxy([stack], '+stack_pred')
# stack-only MAE on holdout
print('stack-only holdout431 MAE: %.3f'%np.abs(stack[va431]-y[va431]).mean())
# hygiene variant: fill all NaN with train medians (simulating harness imputation) vs defined fills
Xdf=m[feats].astype(float)
med=Xdf[tr431].median()
Xh=Xdf.fillna(med).values
Xb=make_bins(Xh); b,tr=fit_gbm(Xb[tr431],y[tr431]); p=gbm_pred(b,tr,Xb[va431])
print('hygiene(median-fill all) MAE %.3f'%np.abs(p-y[va431]).mean())

# ---- cell ----
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

# ---- cell ----
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
SF=['ew_28','ew_56','ew_84','lr_mean28','lr_med28','spend_28_prior','spend_84_prior','active_28','days_since_last','baskets_28','baskets_84','avg_basket_84','n_products_84','win_cv','spend_lag364','spend_lag336']
Zdf=m[SF].astype(float).copy()
for c in SF: Zdf[c]=Zdf[c].fillna(0.0)
Zdf['days_since_last']=Zdf['days_since_last'].clip(0,150)
Z=Zdf.values
def design(Zm):
    q=((Zm[:,0]/100.0)**2)[:,None]; inter=(Zm[:,0]*Zm[:,3]/1e4)[:,None]
    return np.hstack([np.ones((len(Zm),1)), Zm, q, inter])
stack=np.full(len(m), np.nan); meta=[]
t0=time.time()
for s in sdays:
    fit_mask = sd <= s-28; pred_mask = sd == s
    if fit_mask.sum()>500:
        w=irls_mae(design(Z[fit_mask]), y[fit_mask])
        stack[pred_mask]=np.maximum(design(Z[pred_mask])@w, 0)
        meta.append((s, int(fit_mask.sum()), round(float(np.abs(stack[pred_mask]-y[pred_mask]).mean()),1)))
    else:
        stack[pred_mask]=Z[pred_mask,0]
        meta.append((s, 0, round(float(np.abs(stack[pred_mask]-y[pred_mask]).mean()),1)))
print('stack built %.1fs'%(time.time()-t0)); print('per-snap:', meta)
print('stack corr %.3f | stack-only MAE(all) %.2f | holdout431 %.3f'%(np.corrcoef(stack,y)[0,1], np.abs(stack-y).mean(), np.abs(stack[va431]-y[va431]).mean()))
X=m[feats].astype(float).values
def run_both(cols_extra, name):
    Xe=np.hstack([X]+[c[:,None] for c in cols_extra]) if cols_extra else X
    Xb=make_bins(Xe)
    b,tr=fit_gbm(Xb[tr431],y[tr431]); p=gbm_pred(b,tr,Xb[va431]); m1=np.abs(p-y[va431]).mean()
    Oh=np.zeros((len(m),12)); u=sorted(set(sd[tr431]))
    for i,d in enumerate(u): Oh[tr431,i]=(sd[tr431]==d); Oh[va431,i]=(sd[va431]==d)
    Xr=np.hstack([Xe,Oh]); mu=Xr[tr431].mean(0); s2=Xr[tr431].std(0)+1e-9
    Zt=(Xr[tr431]-mu)/s2; Zv=(Xr[va431]-mu)/s2
    w=np.linalg.solve(Zt.T@Zt+3.0*np.eye(Zt.shape[1]), Zt.T@y[tr431])
    m2=np.abs(Zv@w-y[va431]).mean()
    print('%-24s GBM %.3f | ridge %.3f'%(name,m1,m2))
run_both([], 'E011 as-is')
run_both([stack], 'E011+stack')
run_both([np.log1p(stack)], 'E011+log1p(stack)')

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings, time, re
warnings.filterwarnings('ignore')
BIG=10**6
# ---------- demo parsing ----------
def parse_demo(demo):
    d=demo.copy()
    def num(s):
        if pd.isna(s): return np.nan
        m=re.search(r'(\d+)', str(s)); return float(m.group(1)) if m else np.nan
    out=pd.DataFrame(index=d.household_key)
    out['demo_age']=d.classification_1.map(num)
    out['demo_c2']=d.classification_2.astype('category').cat.codes.replace(-1,np.nan)
    out['demo_c3']=d.classification_3.map(num)
    out['demo_c4']=d.classification_4.map(num)
    out['demo_c5']=d.classification_5.map(num)
    out['homeowner']=d.homeowner_desc.astype('category').cat.codes.replace(-1,np.nan)
    out['kid_cat']=d.kid_category_desc.astype('category').cat.codes.replace(-1,np.nan)
    return out
DEMO=parse_demo(A.snapshot().table('demographics'))

# ---------- core builder ----------
def build_core(tx, s, fp, cS0, cB0, cA0, EW, hh_all, basket_tab):
    # fp: first-day per hh (aligned to hh_all); cS0/cB0/cA0: (n_hh, s+2) cumsums with leading zero; EW: dict hl->ew value per hh
    n=len(hh_all)
    def W(a,b):  # sum days a..b inclusive (1-based); a>=1
        a=max(a,1)
        if b<a: return np.zeros(n)
        return cS0[:,b+1]-cS0[:,a]
    def WB(a,b):
        a=max(a,1)
        if b<a: return np.zeros(n)
        return cB0[:,b+1]-cB0[:,a]
    def WA(a,b):
        a=max(a,1)
        if b<a: return np.zeros(n)
        return cA0[:,b+1]-cA0[:,a]
    f=pd.DataFrame(index=hh_all)
    f['spend_7']=W(s-6,s); f['spend_14']=W(s-13,s); f['spend_28']=W(s-27,s)
    f['spend_56']=W(s-55,s); f['spend_84']=W(s-83,s); f['spend_180']=W(s-179,s); f['spend_365']=W(s-364,s)
    f['spend_28_prior']=W(s-55,s-28); f['spend_84_prior']=W(s-167,s-84)
    f['baskets_28']=WB(s-27,s); f['baskets_84']=WB(s-83,s); f['trips_364']=WB(s-363,s)
    f['active_days_28']=WA(s-27,s); f['active_days_364']=WA(s-363,s)
    f['days_since_last']=s-fp.replace(-1,np.nan) if False else s-np.where(fp>0,fp,np.nan)
    f['days_since_first']=s-np.where(fp>0,fp,np.nan)
    f['avg_basket_84']=f['spend_84']/np.maximum(f['baskets_84'],1)
    f['trips_per_wk_84']=f['baskets_84']/12.0
    f['spend_28_ratio']=f['spend_28']/(f['spend_28_prior']+1.0)
    f['active_28']=(f['baskets_28']>0).astype(float)
    for hl in [7,14,28,56,84,180]: f['ew_%d'%hl]=EW[hl]
    # 28d-window stats over trailing 364d
    wins=[]
    for k in range(13):
        b=s-28*k; a=s-28*(k+1)+1
        wins.append(W(a,b) if a>=1 else np.full(n,np.nan))
    Wm=np.vstack(wins)  # 13 x n
    valid=(Wm==Wm) & (np.arange(13)[:,None] <= ((s-fp[None,:])//28))  # window fully in tenure
    V=np.where(valid,Wm,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f['lr_mean28']=np.nanmean(V,axis=0); f['lr_med28']=np.nanmedian(V,axis=0)
        f['lr_std28']=np.nanstd(V,axis=0); f['lr_max28']=np.nanmax(V,axis=0); f['lr_min28']=np.nanmin(V,axis=0)
        f['n_valid_wins']=np.sum(valid,axis=0).astype(float); f['lr_active_wins']=np.nansum((V>0),axis=0).astype(float)
        mu=np.nanmean(V,axis=0); sd=np.nanstd(V,axis=0)
        f['win_cv']=sd/np.where(mu>0,mu,np.nan)
    f['spend_364']=W(s-363,s)
    f['spend_lag336']=np.where(s-364>=1, W(s-363,s-336), np.nan)
    f['spend_lag364']=np.where(s-392>=1, W(s-391,s-364), np.nan)
    f['spend_lag392']=np.where(s-420>=1, W(s-419,s-392), np.nan)
    f['longrun_wk']=f['spend_365']/52.0
    f['ratio28_lr']=f['spend_28']/(4*f['longrun_wk']+1.0)
    f['ratio84_lr']=f['spend_84']/(12*f['longrun_wk']+1.0)
    # basket-level stats over 84d
    bt=basket_tab[(basket_tab.day>s-84)&(basket_tab.day<=s)]
    g=bt.groupby('household_key')['bspend']
    f['basket_max_84']=g.max().reindex(hh_all)
    f['basket_std_84']=g.std().reindex(hh_all)
    f['basket_med_84']=g.median().reindex(hh_all)
    # calendar
    f['snap_day']=float(s); f['wk_of_year']=float(((s+8)//7)%52)
    f['ann_sin']=np.sin(2*np.pi*s/364.0); f['ann_cos']=np.cos(2*np.pi*s/364.0)
    f['win_sin']=np.sin(2*np.pi*(s+14.5)/364.0); f['win_cos']=np.cos(2*np.pi*(s+14.5)/364.0)
    # demo
    for c in ['demo_age','demo_c2','demo_c3','demo_c4','demo_c5','homeowner','kid_cat']:
        f[c]=DEMO[c].reindex(hh_all)
    f['has_demo']=DEMO['demo_age'].reindex(hh_all).notna().astype(float)
    return f

# ---------- stack helpers ----------
def irls(Z,y,iters=15,lam=1e-3):
    w=np.linalg.lstsq(Z,y,rcond=None)[0]
    for _ in range(iters):
        r=y-Z@w; a=np.maximum(np.abs(r),1.0); Wt=1.0/a
        w=np.linalg.solve((Z*Wt[:,None]).T@Z+lam*np.eye(Z.shape[1]), (Z*Wt[:,None]).T@y)
    return w
def logit_irls(Z,y,iters=8,lam=1.0):
    w=np.zeros(Z.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-(Z@w))); Wt=p*(1-p)+1e-6
        g=Z.T@(y-p)-lam*w
        H=(Z*Wt[:,None]).T@Z+lam*np.eye(Z.shape[1])
        w=w+np.linalg.solve(H,g)
    return w

SF=['spend_28','spend_56','spend_84','spend_28_prior','spend_84_prior','spend_364','longrun_wk','ew_28','ew_56','ew_84','lr_mean28','lr_med28','lr_max28','baskets_28','baskets_84','avg_basket_84','trips_per_wk_84','active_days_28','days_since_last','win_cv']
print('setup ok'); print('DEMO rows', len(DEMO))

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
snap=A.snapshot(459); tx=snap.table('transactions').copy()
hh_all=np.sort(tx.household_key.unique()); n=len(hh_all)
fp=tx.groupby('household_key')['day'].min().reindex(hh_all).fillna(0).values.astype(int)
tx['bspend']=tx.groupby('basket_id')['sales_value'].transform('sum')
basket_tab=tx.groupby('basket_id').agg(household_key=('household_key','first'), day=('day','first'), bspend=('bspend','first')).reset_index()
EWG={}
for hl in [7,14,28,56,84,180]:
    r=2.0**(1.0/hl)
    EWG[hl]=tx.assign(v=tx.sales_value*r**tx.day).groupby('household_key')['v'].sum().reindex(hh_all).fillna(0.0).values
demo=snap.table('demographics')
def num(s):
    if pd.isna(s): return np.nan
    import re as _re; m=_re.search(r'(\d+)',str(s)); return float(m.group(1)) if m else np.nan
D=pd.DataFrame(index=demo.household_key)
D['demo_age']=demo.classification_1.map(num); D['demo_c2']=demo.classification_2.astype('category').cat.codes.replace(-1,np.nan)
D['demo_c3']=demo.classification_3.map(num); D['demo_c4']=demo.classification_4.map(num); D['demo_c5']=demo.classification_5.map(num)
D['homeowner']=demo.homeowner_desc.astype('category').cat.codes.replace(-1,np.nan)
D['kid_cat']=demo.kid_category_desc.astype('category').cat.codes.replace(-1,np.nan)
SDAYS=[95,123,151,179,207,235,263,291,319,347,375,403,431,459]
t0=time.time(); parts=[]
for s in SDAYS:
    txs=tx[tx.day<=s]
    cS=np.zeros((n,s+2)); cB=np.zeros((n,s+2)); cA=np.zeros((n,s+2))
    hh_idx=pd.Series(np.arange(n),index=hh_all); gi=txs.household_key.map(hh_idx).values
    np.add.at(cS[:,1:],(gi,txs.day.values),txs.sales_value.values)
    np.add.at(cB[:,1:],(gi,txs.day.values),1.0); np.add.at(cA[:,1:],(gi,txs.day.values),1.0)
    cS=np.cumsum(cS,axis=1); cB=np.cumsum(cB,axis=1); cA=np.cumsum(cA,axis=1)
    f=pd.DataFrame(index=hh_all)
    def W(a,b):
        a=max(a,1); return cS[:,b+1]-cS[:,a] if b>=a else np.zeros(n)
    def WB(a,b):
        a=max(a,1); return cB[:,b+1]-cB[:,a] if b>=a else np.zeros(n)
    def WA(a,b):
        a=max(a,1); return cA[:,b+1]-cA[:,a] if b>=a else np.zeros(n)
    f['spend_7']=W(s-6,s); f['spend_14']=W(s-13,s); f['spend_28']=W(s-27,s)
    f['spend_56']=W(s-55,s); f['spend_84']=W(s-83,s); f['spend_180']=W(s-179,s); f['spend_365']=W(s-364,s)
    f['spend_28_prior']=W(s-55,s-28); f['spend_84_prior']=W(s-167,s-84)
    f['baskets_28']=WB(s-27,s); f['baskets_84']=WB(s-83,s); f['trips_364']=WB(s-363,s)
    f['active_days_28']=WA(s-27,s); f['active_days_364']=WA(s-363,s)
    f['days_since_last']=s-np.where(fp>0,fp,np.nan); f['days_since_first']=s-np.where(fp>0,fp,np.nan)
    f['avg_basket_84']=f['spend_84']/np.maximum(f['baskets_84'],1); f['trips_per_wk_84']=f['baskets_84']/12.0
    f['spend_28_ratio']=f['spend_28']/(f['spend_28_prior']+1.0); f['active_28']=(f['baskets_28']>0).astype(float)
    for hl in [7,14,28,56,84,180]: f['ew_%d'%hl]=EWG[hl]*(2.0**(-s/hl))
    wins=np.vstack([W(s-28*(k+1)+1,s-28*k) if s-28*(k+1)+1>=1 else np.full(n,np.nan) for k in range(13)])
    valid=(wins==wins)&(np.arange(13)[:,None]<=((s-np.where(fp>0,fp,s)[None,:])//28))
    V=np.where(valid,wins,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f['lr_mean28']=np.nanmean(V,axis=0); f['lr_med28']=np.nanmedian(V,axis=0)
        f['lr_std28']=np.nanstd(V,axis=0); f['lr_max28']=np.nanmax(V,axis=0); f['lr_min28']=np.nanmin(V,axis=0)
        f['n_valid_wins']=valid.sum(axis=0).astype(float); f['lr_active_wins']=np.nansum(V>0,axis=0).astype(float)
        mu=np.nanmean(V,axis=0); sdd=np.nanstd(V,axis=0); f['win_cv']=sdd/np.where(mu>0,mu,np.nan)
    f['spend_364']=W(s-363,s)
    f['spend_lag336']=np.where(s-364>=1,W(s-363,s-336),np.nan)
    f['spend_lag364']=np.where(s-392>=1,W(s-391,s-364),np.nan)
    f['spend_lag392']=np.where(s-420>=1,W(s-419,s-392),np.nan)
    f['longrun_wk']=f['spend_365']/52.0
    f['ratio28_lr']=f['spend_28']/(4*f['longrun_wk']+1.0); f['ratio84_lr']=f['spend_84']/(12*f['longrun_wk']+1.0)
    bt=basket_tab[(basket_tab.day>s-84)&(basket_tab.day<=s)]; g=bt.groupby('household_key')['bspend']
    f['basket_max_84']=g.max().reindex(hh_all); f['basket_std_84']=g.std().reindex(hh_all); f['basket_med_84']=g.median().reindex(hh_all)
    f['snap_day']=float(s); f['wk_of_year']=float(((s+8)//7)%52)
    f['ann_sin']=np.sin(2*np.pi*s/364.0); f['ann_cos']=np.cos(2*np.pi*s/364.0)
    f['win_sin']=np.sin(2*np.pi*(s+14.5)/364.0); f['win_cos']=np.cos(2*np.pi*(s+14.5)/364.0)
    for c in D.columns: f[c]=D[c].reindex(hh_all)
    f['has_demo']=D['demo_age'].reindex(hh_all).notna().astype(float)
    f['household_key']=hh_all; f['snapshot_day']=s
    parts.append(f.reset_index(drop=True))
F=pd.concat(parts,ignore_index=True)
print('core built', F.shape, '%.0fs'%(time.time()-t0))
# ---- causal stack ----
def irls(Z,y,iters=15,lam=1e-3):
    w=np.linalg.lstsq(Z,y,rcond=None)[0]
    for _ in range(iters):
        r=y-Z@w; a=np.maximum(np.abs(r),1.0); Wt=1.0/a
        w=np.linalg.solve((Z*Wt[:,None]).T@Z+lam*np.eye(Z.shape[1]),(Z*Wt[:,None]).T@y)
    return w
def logit_irls(Z,y,iters=8,lam=1.0):
    w=np.zeros(Z.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-(Z@w))); Wt=p*(1-p)+1e-6
        w=w+np.linalg.solve((Z*Wt[:,None]).T@Z+lam*np.eye(Z.shape[1]),Z.T@(y-p)-lam*w)
    return w
SF=['spend_28','spend_56','spend_84','spend_28_prior','spend_84_prior','spend_364','longrun_wk','ew_28','ew_56','ew_84','lr_mean28','lr_med28','lr_max28','baskets_28','baskets_84','avg_basket_84','trips_per_wk_84','active_days_28','days_since_last','win_cv']
Zb=F[SF].astype(float).copy()
for c in SF: Zb[c]=Zb[c].fillna(0.0)
Zb['days_since_last']=Zb['days_since_last'].clip(0,150)
Z=Zb.values; y_map=tt.set_index(['household_key','snapshot_day'])['future_spend_4w']
F['y_causal']=F.set_index(['household_key','snapshot_day']).index.map(y_map)
sd_arr=F['snapshot_day'].values; sdays=sorted(F.snapshot_day.unique())
stack=np.full(len(F),np.nan); pzero=np.full(len(F),np.nan)
for s in sdays:
    fm=sd_arr<=s-28; pm=sd_arr==s
    if fm.sum()>500:
        yk=F['y_causal'].values; okf=fm&~np.isnan(yk)
        Zf=Z[okf]; yf=yk[okf]
        w1=irls(Zf,yf); stack[pm]=np.maximum(Z[pm]@w1,0)
        yb=(yf>0).astype(float); w2=logit_irls(Zf,yb); pzero[pm]=1/(1+np.exp(-(Z[pm]@w2)))
    else:
        stack[pm]=Z[pm,7]; pzero[pm]=0.2
F['stack_lin']=stack; F['p_zero']=pzero
print('stack done %.0fs'%(time.time()-t0))
path=A.save_table(F,'core_offline_check'); print(path)
# ---- proxy eval holdout-431 ----
def make_bins(X,nb=32):
    n_,p_=X.shape; Xb=np.empty((n_,p_),dtype=np.int16)
    for j in range(p_):
        x=X[:,j]; ok=~np.isnan(x)
        if ok.sum()<50: Xb[:,j]=nb; continue
        e=np.unique(np.quantile(x[ok],np.linspace(0,1,nb+1)[1:-1]))
        col=np.full(n_,nb,dtype=np.int16); col[ok]=np.searchsorted(e,x[ok],side='right'); Xb[:,j]=col
    return Xb
def fit_gbm(Xb,y,n_trees=200,lr=0.08,depth=4,min_leaf=20,seed=0):
    rng=np.random.RandomState(seed); n_,p_=Xb.shape; base=y.mean(); pred=np.full(n_,base); trees=[]; NB=33
    for it in range(n_trees):
        resid=y-pred; ridx=rng.choice(n_,int(0.8*n_),replace=False); cols=rng.choice(p_,max(5,int(0.8*p_)),replace=False)
        nodes=[]
        def rec(idx,dep):
            i=len(nodes); nodes.append(['leaf',resid[idx].mean()])
            if dep>=depth or len(idx)<2*min_leaf: return i
            best=(None,None,-1e18)
            for j in cols:
                b=Xb[idx,j]; hs=np.bincount(b,weights=resid[idx],minlength=NB).astype(float); hc=np.bincount(b,minlength=NB).astype(float)
                cs=np.cumsum(hs); cc=np.cumsum(hc); cl=cc[:-1]; sl=cs[:-1]; cr=cc[-1]-cl; sr=cs[-1]-sl
                gain=np.where((cl>=min_leaf)&(cr>=min_leaf),sl*sl/(cl+1e-9)+sr*sr/(cr+1e-9),-1e18)
                t=int(np.argmax(gain))
                if gain[t]>best[2]: best=(j,t,gain[t])
            j,t,_=best
            if j is None: return i
            mask=Xb[idx,j]<=t; li,ri=idx[mask],idx[~mask]
            if len(li)<min_leaf or len(ri)<min_leaf: return i
            nodes[i]=['split',j,t,None,None]; nodes[i][3]=rec(li,dep+1); nodes[i][4]=rec(ri,dep+1); return i
        rec(ridx,0); out=np.empty(n_)
        def prd(ni,idx):
            nd=nodes[ni]
            if nd[0]=='leaf': out[idx]=nd[1]; return
            _,j,t,l,r=nd; m=Xb[idx,j]<=t; prd(l,idx[m]); prd(r,idx[~m])
        prd(0,np.arange(n_)); trees.append(nodes); pred+=lr*out
    return base,trees
def gbm_pred(base,trees,Xb,lr=0.08):
    out=np.full(len(Xb),base)
    for nodes in trees:
        o=np.empty(len(Xb))
        def prd(ni,idx):
            nd=nodes[ni]
            if nd[0]=='leaf': o[idx]=nd[1]; return
            _,j,t,l,r=nd; m=Xb[idx,j]<=t; prd(l,idx[m]); prd(r,idx[~m])
        prd(0,np.arange(len(Xb))); out+=lr*o
    return out
mrg=F[(F.snapshot_day<=431)]
ttm=tt.merge(mrg,on=['household_key','snapshot_day'])
feats=[c for c in F.columns if c not in ('household_key','snapshot_day','y_causal')]
Xa=ttm[feats].astype(float).values; ya=ttm['future_spend_4w'].values; sda=ttm['snapshot_day'].values
tr=sda<=403; va=sda==431
Xb=make_bins(Xa); b,tr_=fit_gbm(Xb[tr],ya[tr]); p=gbm_pred(b,tr_,Xb[va])
print('NEW holdout431 MAE %.3f (n_val=%d)'%(np.abs(p-ya[va]).mean(),va.sum()))
Xb2=make_bins(np.hstack([Xa,np.log1p(np.maximum(stack[sda<=431],0))[:,None]]))
b,tr_=fit_gbm(Xb2[tr],ya[tr]); p=gbm_pred(b,tr_,Xb2[va])
print('NEW+logstack MAE %.3f'%np.abs(p-ya[va]).mean())
e11=A.load_saved('e011_discounts.parquet'); ttm2=tt.merge(e11,on=['household_key','snapshot_day'])
f2=[c for c in e11.columns if c not in ('household_key','snapshot_day')]
for c in f2:
    if ttm2[c].dtype==object: ttm2[c]=ttm2[c].astype('category').cat.codes.replace(-1,np.nan)
X2=ttm2[f2].astype(float).values; Xb3=make_bins(X2); b,tr_=fit_gbm(Xb3[tr],ya[tr]); p=gbm_pred(b,tr_,Xb3[va])
print('E011 holdout431 MAE %.3f'%np.abs(p-ya[va]).mean())

# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
F=A.load_saved('core_offline_check')
tt=A.train_targets()
def irls(Z,y,iters=15,lam=1e-3):
    w=np.linalg.lstsq(Z,y,rcond=None)[0]
    for _ in range(iters):
        r=y-Z@w; a=np.maximum(np.abs(r),1.0); Wt=1.0/a
        w=np.linalg.solve((Z*Wt[:,None]).T@Z+lam*np.eye(Z.shape[1]),(Z*Wt[:,None]).T@y)
    return w
def logit_irls(Z,y,iters=8,lam=1.0):
    w=np.zeros(Z.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-(Z@w))); Wt=p*(1-p)+1e-6
        w=w+np.linalg.solve((Z*Wt[:,None]).T@Z+lam*np.eye(Z.shape[1]),Z.T@(y-p)-lam*w)
    return w
SF=['spend_28','spend_56','spend_84','spend_28_prior','spend_84_prior','spend_364','longrun_wk','ew_28','ew_56','ew_84','lr_mean28','lr_med28','lr_max28','baskets_28','baskets_84','avg_basket_84','trips_per_wk_84','active_days_28','days_since_last','win_cv']
Zb=F[SF].astype(float).copy()
for c in SF: Zb[c]=Zb[c].fillna(0.0)
Zb['days_since_last']=Zb['days_since_last'].clip(0,150)
Z=Zb.values
yk_map=tt.set_index(['household_key','snapshot_day'])['future_spend_4w']
yk=F.set_index(['household_key','snapshot_day']).index.map(yk_map).values.astype(float)
sd_arr=F['snapshot_day'].values; sdays=sorted(F.snapshot_day.unique())
stack=np.full(len(F),np.nan); pzero=np.full(len(F),np.nan); meta=[]
for s in sdays:
    fm=sd_arr<=s-28; pm=sd_arr==s
    if fm.sum()>500:
        okf=fm&~np.isnan(yk); Zf=Z[okf]; yf=yk[okf]
        w1=irls(Zf,yf); stack[pm]=np.maximum(Z[pm]@w1,0)
        w2=logit_irls(Zf,(yf>0).astype(float)); pzero[pm]=1/(1+np.exp(-(Z[pm]@w2)))
        meta.append((s,round(float(np.abs(stack[pm]-yk[pm]).mean()),1)))
    else:
        stack[pm]=Z[pm,7]; pzero[pm]=0.2; meta.append((s,-1))
F['stack_lin']=stack; F['p_zero']=pzero
print('per-snap stack MAE:',meta)
path=A.save_table(F,'stack_v1_features')
print('SAVED:',path)
# proxy eval
def make_bins(X,nb=32):
    n_,p_=X.shape; Xb=np.empty((n_,p_),dtype=np.int16)
    for j in range(p_):
        x=X[:,j]; ok=~np.isnan(x)
        if ok.sum()<50: Xb[:,j]=nb; continue
        e=np.unique(np.quantile(x[ok],np.linspace(0,1,nb+1)[1:-1]))
        col=np.full(n_,nb,dtype=np.int16); col[ok]=np.searchsorted(e,x[ok],side='right'); Xb[:,j]=col
    return Xb
def fit_gbm(Xb,y,n_trees=200,lr=0.08,depth=4,min_leaf=20,seed=0):
    rng=np.random.RandomState(seed); n_,p_=Xb.shape; base=y.mean(); pred=np.full(n_,base); trees=[]; NB=33
    for it in range(n_trees):
        resid=y-pred; ridx=rng.choice(n_,int(0.8*n_),replace=False); cols=rng.choice(p_,max(5,int(0.8*p_)),replace=False)
        nodes=[]
        def rec(idx,dep):
            i=len(nodes); nodes.append(['leaf',resid[idx].mean()])
            if dep>=depth or len(idx)<2*min_leaf: return i
            best=(None,None,-1e18)
            for j in cols:
                b=Xb[idx,j]; hs=np.bincount(b,weights=resid[idx],minlength=NB).astype(float); hc=np.bincount(b,minlength=NB).astype(float)
                cs=np.cumsum(hs); cc=np.cumsum(hc); cl=cc[:-1]; sl=cs[:-1]; cr=cc[-1]-cl; sr=cs[-1]-sl
                gain=np.where((cl>=min_leaf)&(cr>=min_leaf),sl*sl/(cl+1e-9)+sr*sr/(cr+1e-9),-1e18)
                t=int(np.argmax(gain))
                if gain[t]>best[2]: best=(j,t,gain[t])
            j,t,_=best
            if j is None: return i
            mask=Xb[idx,j]<=t; li,ri=idx[mask],idx[~mask]
            if len(li)<min_leaf or len(ri)<min_leaf: return i
            nodes[i]=['split',j,t,None,None]; nodes[i][3]=rec(li,dep+1); nodes[i][4]=rec(ri,dep+1); return i
        rec(ridx,0); out=np.empty(n_)
        def prd(ni,idx):
            nd=nodes[ni]
            if nd[0]=='leaf': out[idx]=nd[1]; return
            _,j,t,l,r=nd; m=Xb[idx,j]<=t; prd(l,idx[m]); prd(r,idx[~m])
        prd(0,np.arange(n_)); trees.append(nodes); pred+=lr*out
    return base,trees
def gbm_pred(base,trees,Xb,lr=0.08):
    out=np.full(len(Xb),base)
    for nodes in trees:
        o=np.empty(len(Xb))
        def prd(ni,idx):
            nd=nodes[ni]
            if nd[0]=='leaf': o[idx]=nd[1]; return
            _,j,t,l,r=nd; m=Xb[idx,j]<=t; prd(l,idx[m]); prd(r,idx[~m])
        prd(0,np.arange(len(Xb))); out+=lr*o
    return out
mrg=F[F.snapshot_day<=431]
ttm=tt.merge(mrg,on=['household_key','snapshot_day'])
feats=[c for c in F.columns if c not in ('household_key','snapshot_day','y_causal')]
Xa=ttm[feats].astype(float).values; ya=ttm['future_spend_4w'].values; sda=ttm['snapshot_day'].values
tr=sda<=403; va=sda==431
Xb=make_bins(Xa); b,tr_=fit_gbm(Xb[tr],ya[tr]); p=gbm_pred(b,tr_,Xb[va])
print('NEW holdout431 MAE %.3f (n_val=%d)'%(np.abs(p-ya[va]).mean(),va.sum()))
e11=A.load_saved('e011_discounts.parquet'); ttm2=tt.merge(e11,on=['household_key','snapshot_day'])
f2=[c for c in e11.columns if c not in ('household_key','snapshot_day')]
for c in f2:
    if ttm2[c].dtype==object: ttm2[c]=ttm2[c].astype('category').cat.codes.replace(-1,np.nan)
X2=ttm2[f2].astype(float).values; Xb3=make_bins(X2); b,tr_=fit_gbm(Xb3[tr],ya[tr]); p=gbm_pred(b,tr_,Xb3[va])
print('E011 holdout431 MAE %.3f'%np.abs(p-ya[va]).mean())