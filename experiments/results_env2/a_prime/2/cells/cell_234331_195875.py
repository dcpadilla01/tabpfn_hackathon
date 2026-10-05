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