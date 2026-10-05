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