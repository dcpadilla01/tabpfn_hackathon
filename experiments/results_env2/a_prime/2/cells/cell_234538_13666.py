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