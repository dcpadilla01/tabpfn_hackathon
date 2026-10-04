import agent_api as api
import pandas as pd, numpy as np

def feats(view, s):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions
    if len(tx)==0:
        return pd.DataFrame(index=hh)
    tx = tx[['household_key','day','basket_id','sales_value','store_id','product_id','trans_time']]
    out = {}
    g = tx.groupby('household_key')
    first_day = g.day.min(); last_day = g.day.max()
    spend_total = g.sales_value.sum()
    out['tenure'] = (s - first_day).reindex(hh).fillna(0)
    out['days_since_last'] = (s - last_day).reindex(hh).fillna(999)
    out['spend_total'] = spend_total.reindex(hh).fillna(0)
    def win(lo, hi):
        m = tx[(tx.day>lo)&(tx.day<=hi)]
        gb = m.groupby('household_key')
        sp = gb.sales_value.sum()
        trips = gb.basket_id.nunique()
        dact = gb.day.nunique()
        st = gb.store_id.nunique()
        pr = gb.product_id.nunique()
        return sp, trips, dact, st, pr
    W = {}
    for k in [1,2,3,4,5,6,13]:
        hi = s-28*(k-1); lo = hi-28
        W[k] = win(lo,hi)
    for k in [1,2,3]:
        sp,tr,da,stt,pr = W[k]
        out[f'nspend_l{k}']=sp.reindex(hh).fillna(0)
        out[f'ntrips_l{k}']=tr.reindex(hh).fillna(0)
        out[f'ndact_l{k}']=da.reindex(hh).fillna(0)
        out[f'nstores_l{k}']=stt.reindex(hh).fillna(0)
        out[f'nprods_l{k}']=pr.reindex(hh).fillna(0)
    for k in [4,5,6,13]:
        out[f'nspend_l{k}']=W[k][0].reindex(hh).fillna(0)
    # evening share 84d
    m84 = tx[(tx.day>s-84)&(tx.day<=s)]
    tt = pd.to_numeric(m84.trans_time, errors='coerce').fillna(0)
    ev = m84.sales_value.where(tt>=1700, 0).groupby(m84.household_key).sum()
    tot = m84.groupby('household_key').sales_value.sum()
    out['nevening_share84'] = (ev/tot.replace(0,np.nan)).reindex(hh)
    # ewm weekly spend levels
    wk = ((tx.day+8)//7).astype(int)
    pv = tx.assign(w=wk).pivot_table(index='w', columns='household_key', values='sales_value', aggfunc='sum').fillna(0)
    wmax = (s+8)//7
    allw = pd.RangeIndex(1, wmax+1)
    pv = pv.reindex(allw).fillna(0)
    ew = pv.ewm(halflife=8, adjust=False).mean().iloc[-1]
    ew4 = pv.ewm(halflife=4, adjust=False).mean().iloc[-1]
    ew13 = pv.ewm(halflife=13, adjust=False).mean().iloc[-1]
    out['newm8']=ew.reindex(hh).fillna(0)
    out['newm4']=ew4.reindex(hh).fillna(0)
    out['newm13']=ew13.reindex(hh).fillna(0)
    df = pd.DataFrame(out)
    # derived
    s123 = df.nspend_l1+df.nspend_l2+df.nspend_l3
    s456 = df.nspend_l4+df.nspend_l5+df.nspend_l6
    d = {}
    d['nf_lspend123']=np.log1p(s123); d['nf_lspend1']=np.log1p(df.nspend_l1)
    d['nf_lspend2']=np.log1p(df.nspend_l2); d['nf_lspend3']=np.log1p(df.nspend_l3)
    d['nf_lspend456']=np.log1p(s456); d['nf_lspend13']=np.log1p(df.nspend_l13)
    d['nf_lspendtot']=np.log1p(df.spend_total)
    d['nf_pow75']=np.power(1+s123,0.75); d['nf_pow90']=np.power(1+s123,0.9); d['nf_pow50']=np.sqrt(1+s123)
    d['nf_lspend123_sq']=d['nf_lspend123']**2
    d['nf_ltrips1']=np.log1p(df.ntrips_l1); d['nf_ltrips123']=np.log1p(df.ntrips_l1+df.ntrips_l2+df.ntrips_l3)
    d['nf_lprods1']=np.log1p(df.nprods_l1); d['nf_ldact1']=np.log1p(df.ndact_l1)
    d['nf_lstores1']=np.log1p(df.nstores_l1)
    d['nf_ldsl']=np.log1p(df.days_since_last.clip(upper=365)); d['nf_ltenure']=np.log1p(df.tenure)
    d['nf_ratio1v2']=df.nspend_l1/(df.nspend_l2+5); d['nf_ratio123v456']=(s123+5)/(s456+5)
    d['nf_ratio1v13']=df.nspend_l1/(df.nspend_l13/13+5)
    d['nf_zero_l1']=(df.nspend_l1<=0).astype(float); d['nf_zero_l2']=((df.nspend_l1+df.nspend_l2)<=0).astype(float)
    d['nf_act1']=df.ndact_l1/28.0
    d['nf_lewm8']=np.log1p(df.newm8); d['nf_lewm4']=np.log1p(df.newm4); d['nf_lewm13']=np.log1p(df.newm13)
    d['nf_pow75_ewm']=np.power(1+df.newm8,0.75)
    d['nf_lavgbasket1']=np.log1p(df.nspend_l1/df.ntrips_l1.replace(0,np.nan))
    wk_of = (s+8)//7
    for h in [1,2]:
        d[f'nf_wsin{h}']=np.sin(2*np.pi*h*wk_of/52.0); d[f'nf_wcos{h}']=np.cos(2*np.pi*h*wk_of/52.0)
    d['nf_ndact123']=np.log1p(df.ndact_l1+df.ndact_l2+df.ndact_l3)
    d['nf_nevening84']=df.nevening_share84
    res = pd.DataFrame(d).reindex(hh)
    return res

tab = api.build_features(feats)
print(tab.shape)
print(tab.columns.tolist())
path = api.save_table(tab, 'nf_transforms.parquet')
print(path)

# offline univariate screen of nf features
t = api.train_targets()
KEY=['household_key','snapshot_day']
def matrix(df):
    m = t.merge(df, on=KEY, how='left')
    yv = m.future_spend_4w.values.astype(float)
    d = m.snapshot_day.values
    X = m.drop(columns=KEY+['future_spend_4w'])
    cols=[]
    for c in X.columns:
        s=X[c]
        if s.dtype==object or str(s.dtype).startswith('category') or s.dtype==bool:
            v=pd.factorize(s)[0].astype(float)
        else:
            v=pd.to_numeric(s,errors='coerce').values.astype(float)
        cols.append(v)
    return np.column_stack(cols), yv, d, list(X.columns)
def ridge_eval(X,y,days,lams=(0.01,0.1,1,10,100)):
    fit=days<=347; val=days>=375
    Xf=X[fit]; mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0); sd[sd<1e-9]=1
    Z=np.where(np.isfinite(X),(X-mu)/sd,0.0)
    Zf=np.hstack([Z[fit],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val],np.ones((val.sum(),1))])
    p=Zf.shape[1]; A=Zf.T@Zf; b=Zf.T@y[fit]; out=[]
    for lam in lams:
        w=np.linalg.solve(A+lam*np.eye(p),b); out.append((np.abs(Zv@w-y[val]).mean(),lam))
    out.sort(); return out[0]
X,yv,d,cols = matrix(tab)
res=[]
for j,c in enumerate(cols):
    mae,lam=ridge_eval(X[:,[j]],yv,d); res.append((mae,c))
res.sort()
for mae,c in res[:25]: print('  %.2f %s'%(mae,c))