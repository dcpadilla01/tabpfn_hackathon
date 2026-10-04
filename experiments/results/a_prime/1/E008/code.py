import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
y = t.future_spend_4w.values.astype(float)
print("targets:", t.shape, "zero share %.3f" % (y==0).mean())
print(t.future_spend_4w.describe(percentiles=[.25,.5,.75,.9,.95,.99]).round(1))
zz = pd.Series(y==0).groupby(t.snapshot_day).mean().round(3)
gm = t.groupby('snapshot_day').future_spend_4w.agg(['mean','median']).round(1)
gm['zero']=zz
print(gm)

tables = {
 'E000': api.baseline_features(),
 'E001': api.load_saved('e001_txhist.parquet'),
 'E002': api.load_saved('e002_channel.parquet'),
 'E003': api.load_saved('e003_catmix.parquet'),
 'E004': api.load_saved('e004_mkt.parquet'),
 'E006': api.load_saved('e006_catmix_mkt.parquet'),
 'E007': api.load_saved('e007_logratio.parquet'),
}
for k,v in tables.items(): print(k, v.shape)
print('E001 cols:', ','.join(map(str,tables['E001'].columns)))
e3x = [c for c in tables['E003'].columns if c not in tables['E001'].columns]
print('E003 extra:', ','.join(map(str,e3x)))

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
    Xm = np.column_stack(cols) if cols else np.zeros((len(yv),0))
    return Xm, yv, d, list(X.columns)

def ridge_eval(X, y, days, lams=(0.01,0.1,1,10,100)):
    fit = days<=347; val = days>=375
    Xf=X[fit]
    mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0); sd[sd<1e-9]=1
    Z=np.where(np.isfinite(X),(X-mu)/sd,0.0)
    Zf=np.hstack([Z[fit],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val],np.ones((val.sum(),1))])
    p=Zf.shape[1]; A=Zf.T@Zf; b=Zf.T@y[fit]
    out=[]
    for lam in lams:
        w=np.linalg.solve(A+lam*np.eye(p),b)
        out.append((np.abs(Zv@w-y[val]).mean(),lam))
    out.sort(); return out[0]

med = np.median(y[(t.snapshot_day<=347).values])
print('baseline fit-median -> offline MAE %.2f' % np.abs(med - y[(t.snapshot_day>=375).values]).mean())

for k in ['E000','E001','E002','E003','E004','E006','E007']:
    X,yv,d,cols = matrix(tables[k])
    mae,lam = ridge_eval(X,yv,d)
    print(k, X.shape, 'ridge offline MAE %.2f (lam %g)'%(mae,lam))

# univariate screen on E003 (contains E001 features too)
X,yv,d,cols = matrix(tables['E003'])
res=[]
for j,c in enumerate(cols):
    mae,lam = ridge_eval(X[:,[j]], yv, d)
    res.append((mae,c))
res.sort()
print('Top univariate (E003):')
for mae,c in res[:18]: print('  %.2f %s'%(mae,c))

# ---- cell ----
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

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
KEY=['household_key','snapshot_day']
E3 = api.load_saved('e003_catmix.parquet')
NF = api.load_saved('nf_transforms.parquet')

m = t.merge(E3[['household_key','snapshot_day','spend_l1','spend_l2','spend_l3','spend_l123_mean']], on=KEY, how='left')
m = m.merge(NF[['household_key','snapshot_day','nf_lspend123','nf_lspend1']], on=KEY, how='left')
print("rows", len(m))
for c in ['spend_l1','spend_l2','spend_l3','spend_l123_mean','nf_lspend1','nf_lspend123']:
    v=m[c]; print(c, "nan:",v.isna().sum(), "zero:",(v==0).sum(), "med:",np.nanmedian(v))

# is spend_l123_mean the mean or sum?
chk = (m.spend_l1+m.spend_l2+m.spend_l3)
print("corr mean vs sum/3:", np.corrcoef(m.spend_l123_mean, chk/3)[0,1])

lg_mean = np.log1p(m.spend_l123_mean)
lg_sum  = np.log1p(chk)
print("corr lg_mean vs lg_sum:", np.corrcoef(lg_mean, lg_sum)[0,1])
print("corr lg_mean vs nf_lspend123:", np.corrcoef(lg_mean, m.nf_lspend123)[0,1])

def uni(col_vals, y, days):
    fit=days<=347; val=days>=375
    x=col_vals.astype(float)
    mu=np.nanmean(x[fit]); sd=np.nanstd(x[fit]); sd=sd if sd>1e-9 else 1
    Z=np.where(np.isfinite(x),(x-mu)/sd,0.0)
    Zf=np.hstack([Z[fit,None],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val,None],np.ones((val.sum(),1))])
    A=Zf.T@Zf; b=Zf.T@y[fit]; p=2
    best=[]
    for lam in [0.01,0.1,1,10,100]:
        w=np.linalg.solve(A+lam*np.eye(p),b); best.append((np.abs(Zv@w-y[val]).mean(),lam))
    best.sort(); return best[0]

y=m.future_spend_4w.values.astype(float); d=m.snapshot_day.values
print("uni log1p(E3 mean):", uni(lg_mean.values,y,d))
print("uni log1p(sum)    :", uni(lg_sum.values,y,d))
print("uni nf_lspend123  :", uni(m.nf_lspend123.values,y,d))
print("uni nf_lspend1    :", uni(m.nf_lspend1.values,y,d))
print("uni log1p(E3 l1)  :", uni(np.log1p(m.spend_l1).values,y,d))
# where do they differ?
diff = (np.abs(lg_sum - m.nf_lspend123) > 0.01) & lg_sum.notna() & m.nf_lspend123.notna()
print("n differing rows:", diff.sum())
if diff.sum():
    print(m.loc[diff, ['household_key','snapshot_day','spend_l1','spend_l2','spend_l3','nf_lspend1','nf_lspend123']].head(10))
# NaN handling: rows where E3 spend_l123_mean is NaN
nn = m.spend_l123_mean.isna()
print("E3 mean NaN rows:", nn.sum(), " their nf_lspend123 values:", m.loc[nn,'nf_lspend123'].unique()[:5], " mean y:", y[nn.values].mean())

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

def feats(view, s):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions
    if len(tx)==0:
        return pd.DataFrame(index=hh)
    tx = tx[['household_key','day','basket_id','sales_value','store_id','product_id','quantity','trans_time']]
    out = {}
    def win(lo,hi): return tx[(tx.day>lo)&(tx.day<=hi)]
    # mid-term lags 7-12 (household seasonality trajectory toward lag-13)
    for k in [7,8,9,10,11,12]:
        hi = s-28*(k-1)
        out[f'nspend_l{k}'] = win(hi-28,hi).groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    # micro windows
    m7, m14, m28 = win(s-7,s), win(s-14,s), win(s-28,s)
    out['nspend7']=m7.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['nspend14']=m14.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['ntrips7']=m7.groupby('household_key').basket_id.nunique().reindex(hh).fillna(0)
    out['nspend28']=m28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
    out['ntrips28']=m28.groupby('household_key').basket_id.nunique().reindex(hh).fillna(0)
    out['ndact28']=m28.groupby('household_key').day.nunique().reindex(hh).fillna(0)
    # 84d basket/volume stats
    m84 = win(s-84,s)
    bs = m84.groupby(['household_key','basket_id']).sales_value.sum().reset_index()
    bst = bs.groupby('household_key').sales_value.agg(['mean','max','std'])
    out['nbask_mean84']=bst['mean'].reindex(hh)
    out['nbask_max84']=bst['max'].reindex(hh)
    out['nbask_std84']=bst['std'].reindex(hh)
    out['ntrips84']=m84.groupby('household_key').basket_id.nunique().reindex(hh).fillna(0)
    out['nunits84']=m84.groupby('household_key').quantity.sum().reindex(hh).fillna(0)
    out['nprods84']=m84.groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    out['nstores84']=m84.groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    # trip gaps in 84d
    dd = m84.groupby('household_key').day.apply(lambda x: np.sort(x.unique()))
    gp = dd.apply(lambda a: np.diff(a))
    out['ngap_mean']=gp.apply(lambda a: a.mean() if len(a)>0 else np.nan).reindex(hh)
    out['ngap_max']=gp.apply(lambda a: a.max() if len(a)>0 else np.nan).reindex(hh)
    out['ngap_std']=gp.apply(lambda a: a.std() if len(a)>1 else np.nan).reindex(hh)
    # day-of-week spend shares (84d)
    sp_dow = m84.assign(dow=m84.day % 7).groupby(['household_key','dow']).sales_value.sum().unstack(fill_value=0)
    tot = sp_dow.sum(axis=1).replace(0,np.nan)
    for dw in range(7):
        col = sp_dow[dw] if dw in sp_dow.columns else pd.Series(0,index=sp_dow.index)
        out[f'ndow{dw}'] = (col/tot).reindex(hh)
    # weekly pivot: EWM levels + volatility
    wk = ((tx.day+8)//7).astype(int)
    pv = tx.assign(w=wk).pivot_table(index='w', columns='household_key', values='sales_value', aggfunc='sum').fillna(0)
    wmax = (s+8)//7
    pv = pv.reindex(range(1,wmax+1)).fillna(0)
    for hl in [4,8,13]:
        out[f'newm{hl}']=pv.ewm(halflife=hl, adjust=False).mean().iloc[-1].reindex(hh).fillna(0)
    l12, l26 = pv.tail(12), pv.tail(26)
    out['nwmean12']=l12.mean(axis=0).reindex(hh).fillna(0)
    out['nwstd12']=l12.std(axis=0).reindex(hh).fillna(0)
    out['nwstd26']=l26.std(axis=0).reindex(hh).fillna(0)
    out['nwmax12']=l12.max(axis=0).reindex(hh).fillna(0)
    out['nwmin12']=l12.min(axis=0).reindex(hh).fillna(0)
    df = pd.DataFrame(out).reindex(hh)
    # derived
    d = {}
    d['nf_pow90_ewm8']=np.power(1+df.newm8,0.9); d['nf_pow90_ewm4']=np.power(1+df.newm4,0.9)
    d['nf_pow90_ewm13']=np.power(1+df.newm13,0.9)
    d['nf_ncv12']=df.nwstd12/df.nwmean12.replace(0,np.nan)
    d['nf_ncv26']=df.nwstd26/df.nwmean12.replace(0,np.nan)
    d['nf_nperactday']=df.nspend28/df.ndact28.replace(0,np.nan)
    d['nf_npertrip28']=df.nspend28/df.ntrips28.replace(0,np.nan)
    d['nf_npertrip84']=df.nbask_mean84
    d['nf_nratio7_28']=df.nspend7/(df.nspend28/4+1)
    d['nf_nratio14_28']=df.nspend14/(df.nspend28/2+1)
    d['nf_nunits_per_trip']=df.nunits84/df.ntrips84.replace(0,np.nan)
    d['nf_nmax_share12']=df.nwmax12/df.nspend28.replace(0,np.nan)
    d['nf_nspend28_pow90']=np.power(1+df.nspend28,0.9)
    d['nf_nspend28_pow75']=np.power(1+df.nspend28,0.75)
    res = pd.concat([df, pd.DataFrame(d)], axis=1).reindex(hh)
    return res

tab = api.build_features(feats)
print(tab.shape)
path = api.save_table(tab, 'nf_candidates.parquet')
print(path)

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
KEY=['household_key','snapshot_day']
def prep(df):
    m = t.merge(df, on=KEY, how='left')
    yv = m.future_spend_4w.values.astype(float); d = m.snapshot_day.values
    X = m.drop(columns=KEY+['future_spend_4w'])
    cols=[]
    for c in X.columns:
        s=X[c]
        if s.dtype==object or str(s.dtype).startswith('category') or s.dtype==bool:
            cols.append(pd.factorize(s)[0].astype(float))
        else:
            cols.append(pd.to_numeric(s,errors='coerce').values.astype(float))
    return np.column_stack(cols), yv, d, list(X.columns)
def ridge_eval(X,y,days,lams=(0.01,0.1,1,10,100),fitmax=347):
    fit=days<=fitmax; val=days>=375
    Xf=X[fit]; mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0); sd[sd<1e-9]=1
    Z=np.where(np.isfinite(X),(X-mu)/sd,0.0)
    Zf=np.hstack([Z[fit],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val],np.ones((val.sum(),1))])
    p=Zf.shape[1]; A=Zf.T@Zf; b=Zf.T@y[fit]; out=[]
    for lam in lams:
        w=np.linalg.solve(A+lam*np.eye(p),b); out.append((np.abs(Zv@w-y[val]).mean(),lam))
    out.sort(); return out[0]

E3 = api.load_saved('e003_catmix.parquet')
NF = api.load_saved('nf_candidates.parquet')
Xb,yb,db,cols_b = prep(E3)
base_mae = ridge_eval(Xb,yb,db)[0]
print("E003 base offline MAE: %.2f"%base_mae)

Xn,yn,dn,cols_n = prep(NF)
# univariate screen
res=[]
for j,c in enumerate(cols_n):
    mae,lam = ridge_eval(Xn[:,[j]],yn,dn); res.append((mae,c))
res.sort()
print("Top univariate (candidates):")
for mae,c in res[:20]: print("  %.2f %s"%(mae,c))
print("Worst:")
for mae,c in res[-8:]: print("  %.2f %s"%(mae,c))

# incremental: add each candidate (one at a time) to E003 base
print("\nIncremental over E003 (top 20):")
inc=[]
for j,c in enumerate(cols_n):
    Xc = np.hstack([Xb, Xn[:,[j]]])
    mae,lam = ridge_eval(Xc,yb,db); inc.append((mae,c))
inc.sort()
for mae,c in inc[:20]: print("  %.3f %s (delta %+.3f)"%(mae,c,mae-base_mae))

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
KEY=['household_key','snapshot_day']
def prep(df):
    m = t.merge(df, on=KEY, how='left')
    yv = m.future_spend_4w.values.astype(float); d = m.snapshot_day.values
    X = m.drop(columns=KEY+['future_spend_4w'])
    cols=[]
    for c in X.columns:
        s=X[c]
        if s.dtype==object or str(s.dtype).startswith('category') or s.dtype==bool:
            cols.append(pd.factorize(s)[0].astype(float))
        else:
            cols.append(pd.to_numeric(s,errors='coerce').values.astype(float))
    return np.column_stack(cols), yv, d, list(X.columns)
def ridge_eval(X,y,days,lams=(0.01,0.1,1,10,100),fitmax=347):
    fit=days<=fitmax; val=days>=375
    Xf=X[fit]; mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0); sd[sd<1e-9]=1
    Z=np.where(np.isfinite(X),(X-mu)/sd,0.0)
    Zf=np.hstack([Z[fit],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val],np.ones((val.sum(),1))])
    p=Zf.shape[1]; A=Zf.T@Zf; b=Zf.T@y[fit]; out=[]
    for lam in lams:
        w=np.linalg.solve(A+lam*np.eye(p),b); out.append((np.abs(Zv@w-y[val]).mean(),lam))
    out.sort(); return out[0]

E3 = api.load_saved('e003_catmix.parquet')
NF = api.load_saved('nf_candidates.parquet')
Xb,yb,db,cols_b = prep(E3)
Xn,yn,dn,cols_n = prep(NF)
base = ridge_eval(Xb,yb,db)[0]
print("base %.3f"%base)

# (a) greedy forward selection of candidate additions
chosen=[]; cur=Xb; cur_mae=base
pool = ['nwmax12','nbask_max84','nwstd12','nspend_l12','nspend7','nf_ncv12','nf_nspend28_pow90','nunits84','nprods84','nf_pow90_ewm4','ngap_std','nspend_l11']
for it in range(6):
    best=None
    for c in pool:
        if c in chosen: continue
        j = cols_n.index(c)
        Xc = np.hstack([cur, Xn[:,[j]]])
        mae,_ = ridge_eval(Xc,yb,db)
        if best is None or mae<best[0]: best=(mae,c,j)
    if best[0] < cur_mae - 1e-4:
        cur_mae, c, j = best; chosen.append(c); cur = np.hstack([cur, Xn[:,[j]]])
        print("greedy +%-18s -> %.3f"%(c,cur_mae))
    else:
        print("no improvement; stop"); break
print("chosen:", chosen)

# (b) pruning: drop weakest E003 features by univariate
uni=[]
for j,c in enumerate(cols_b):
    mae,_=ridge_eval(Xb[:,[j]],yb,db); uni.append((mae,c,j))
uni.sort()
weak = [c for _,c,_ in uni[-15:]]
keep_idx = [j for _,_,j in uni if _ not in [u[0] for u in uni[-15:]]]
Xp = Xb[:, [j for _,_,j in uni[:-15]]]
print("pruned-15 offline: %.3f"%ridge_eval(Xp,yb,db)[0])
Xp2 = Xb[:, [j for _,_,j in uni[:-8]]]
print("pruned-8  offline: %.3f"%ridge_eval(Xp2,yb,db)[0])

# (c) zero/intermittency composites built from E001 raw windows
E1 = api.load_saved('e001_txhist.parquet')
m = t.merge(E1[['household_key','snapshot_day','spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','spend_l13','zero_recent','spend_l123_mean','spend_l456_mean','momentum']], on=KEY, how='left')
zz = {}
zz['n_zerowin_l6'] = ((m[['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6']]<=0).sum(axis=1)).astype(float)
zz['n_zerowin_l3'] = ((m[['spend_l1','spend_l2','spend_l3']]<=0).sum(axis=1)).astype(float)
zz['nf_blend'] = 0.5*m.spend_l123_mean + 0.3*m.spend_l456_mean + 0.2*m.spend_l13/13*13
zz['nf_blend2'] = 0.6*m.spend_l123_mean + 0.4*m.spend_l456_mean
zz['nf_min_l3'] = m[['spend_l1','spend_l2','spend_l3']].min(axis=1)
zz['nf_max_l3'] = m[['spend_l1','spend_l2','spend_l3']].max(axis=1)
zz['nf_std_l3'] = m[['spend_l1','spend_l2','spend_l3']].std(axis=1)
zz['nf_zerowin_x_spend'] = zz['n_zerowin_l6']*m.spend_l123_mean
zz['nf_mom_x_spend'] = m.momentum*m.spend_l123_mean
zz['nf_pow90_blend'] = np.power(1+zz['nf_blend'],0.9)
Z = pd.DataFrame(zz)
Xz = Z.values.astype(float)
print("\nZero/intermittency composites (univariate + incremental over E003):")
for k,c in enumerate(zz):
    u = ridge_eval(Xz[:,[k]],yb,db)[0]
    i = ridge_eval(np.hstack([Xb,Xz[:,[k]]]),yb,db)[0]
    print("  %-22s uni %7.2f  incr %7.3f (delta %+.3f)"%(c,u,i,i-base))

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
KEY=['household_key','snapshot_day']
def prep(df):
    m = t.merge(df, on=KEY, how='left')
    yv = m.future_spend_4w.values.astype(float); d = m.snapshot_day.values
    X = m.drop(columns=KEY+['future_spend_4w'])
    cols=[]
    for c in X.columns:
        s=X[c]
        if s.dtype==object or str(s.dtype).startswith('category') or s.dtype==bool:
            cols.append(pd.factorize(s)[0].astype(float))
        else:
            cols.append(pd.to_numeric(s,errors='coerce').values.astype(float))
    return np.column_stack(cols), yv, d, list(X.columns)
def ridge_eval(X,y,days,lams=(0.01,0.1,1,10,100),fitmax=347):
    fit=days<=fitmax; val=days>=375
    Xf=X[fit]; mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0); sd[sd<1e-9]=1
    Z=np.where(np.isfinite(X),(X-mu)/sd,0.0)
    Zf=np.hstack([Z[fit],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val],np.ones((val.sum(),1))])
    p=Zf.shape[1]; A=Zf.T@Zf; b=Zf.T@y[fit]; out=[]
    for lam in lams:
        w=np.linalg.solve(A+lam*np.eye(p),b); out.append((np.abs(Zv@w-y[val]).mean(),lam))
    out.sort(); return out[0]

E3 = api.load_saved('e003_catmix.parquet')
NF = api.load_saved('nf_candidates.parquet')
E1 = api.load_saved('e001_txhist.parquet')
Xb,yb,db,cols_b = prep(E3)
Xn,yn,dn,cols_n = prep(NF)
base = ridge_eval(Xb,yb,db)[0]

greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
Xg = np.hstack([Xb]+[Xn[:,[cols_n.index(c)]] for c in greedy])
g_mae = ridge_eval(Xg,yb,db)[0]
print("E003+greedy6: %.3f"%g_mae)

# add remaining pool features one at a time on top of greedy
pool2 = ['nspend_l11','nspend_l10','nspend_l9','nspend_l8','nspend_l7','nf_ncv12','nbask_max84','nwstd12','nprods84','nf_nspend28_pow75','newm4','nspend14','ngap_mean','nbask_mean84','ntrips28','ndact28']
inc=[]
for c in pool2:
    Xc = np.hstack([Xg, Xn[:,[cols_n.index(c)]]])
    inc.append((ridge_eval(Xc,yb,db)[0],c))
inc.sort()
for mae,c in inc[:8]: print("  +%-16s %.3f (delta %+.3f)"%(c,mae,mae-g_mae))

# population-split hinges from E001
m = t.merge(E1[['household_key','snapshot_day','spend_l1','spend_l2','spend_l3','zero_recent']], on=KEY, how='left')
s123 = m.spend_l1+m.spend_l2+m.spend_l3
act = (m.zero_recent==0).astype(float).values
inact = 1-act
H = pd.DataFrame({
 'h_act_s123': s123.values*act,
 'h_inact_s123': s123.values*inact,
 'h_act_l1': m.spend_l1.values*act,
 'h_inact_l1': m.spend_l1.values*inact,
 'h_act_pow90': np.power(1+s123.values,0.9)*act,
 'h_inact_pow90': np.power(1+s123.values,0.9)*inact,
 'h_inact_flag': inact,
})
Xh = H.values.astype(float)
for k,c in enumerate(H.columns):
    i = ridge_eval(np.hstack([Xg,Xh[:,[k]]]),yb,db)[0]
    print("  hinge %-16s incr %.3f (delta %+.3f)"%(c,i,i-g_mae))
# all hinges together
print("  all hinges      incr %.3f (delta %+.3f)"%(ridge_eval(np.hstack([Xg,Xh]),yb,db)[0], ridge_eval(np.hstack([Xg,Xh]),yb,db)[0]-g_mae))

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

def feats(view, s):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions
    out = {}
    # population seasonal uplift: avg weekly total spend at same week_no one year earlier
    if len(tx)>0:
        wk = ((tx.day+8)//7).astype(int)
        wtot = tx.assign(w=wk).groupby('w').sales_value.sum()
        wmax = (s+8)//7
        nxt = sorted(set(range(wmax+1, wmax+5)))
        vals=[]
        for w in nxt:
            cands = [w-52, w-104]
            cands = [c for c in cands if c>=1 and c in wtot.index]
            vals.append(wtot[cands].mean() if cands else np.nan)
        overall = wtot[wmax-4:wmax+1].mean() if wmax>=5 else wtot.mean()
        out['nseas_uplift'] = np.nansum(vals)/ (overall if overall>0 else np.nan)
        out['nseas_uplift_mean'] = np.nanmean(vals)/(overall if overall>0 else np.nan)
    else:
        out['nseas_uplift']=np.nan; out['nseas_uplift_mean']=np.nan
    # per-household same-weeks-last-year spend (l13) and ratio to recent
    if len(tx)>0:
        m = tx[(tx.day>s-364)&(tx.day<=s-336)]
        out['nspend_ly'] = m.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
        m28 = tx[(tx.day>s-28)&(tx.day<=s)]
        sp28 = m28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0)
        out['nratio_ly'] = sp28/(out['nspend_ly']/13.0 + 5)
    else:
        out['nspend_ly']=0.0; out['nratio_ly']=np.nan
    df = pd.DataFrame(out).reindex(hh)
    return df

tab = api.build_features(feats)
print(tab.shape, tab.columns.tolist())
print(tab[['nseas_uplift','nseas_uplift_mean']].describe().round(3))
print(tab.groupby('snapshot_day')[['nseas_uplift','nseas_uplift_mean']].mean().round(3))
path = api.save_table(tab, 'nf_seasonal.parquet')
print(path)

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
KEY=['household_key','snapshot_day']
def prep(df):
    m = t.merge(df, on=KEY, how='left')
    yv = m.future_spend_4w.values.astype(float); d = m.snapshot_day.values
    X = m.drop(columns=KEY+['future_spend_4w'])
    cols=[]
    for c in X.columns:
        s=X[c]
        if s.dtype==object or str(s.dtype).startswith('category') or s.dtype==bool:
            cols.append(pd.factorize(s)[0].astype(float))
        else:
            cols.append(pd.to_numeric(s,errors='coerce').values.astype(float))
    return np.column_stack(cols), yv, d, list(X.columns)
def ridge_eval(X,y,days,lams=(0.01,0.1,1,10,100),fitmax=347):
    fit=days<=fitmax; val=days>=375
    Xf=X[fit]; mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0); sd[sd<1e-9]=1
    Z=np.where(np.isfinite(X),(X-mu)/sd,0.0)
    Zf=np.hstack([Z[fit],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val],np.ones((val.sum(),1))])
    p=Zf.shape[1]; A=Zf.T@Zf; b=Zf.T@y[fit]; out=[]
    for lam in lams:
        w=np.linalg.solve(A+lam*np.eye(p),b); out.append((np.abs(Zv@w-y[val]).mean(),lam))
    out.sort(); return out[0]

E3 = api.load_saved('e003_catmix.parquet')
NF = api.load_saved('nf_candidates.parquet')
SE = api.load_saved('nf_seasonal.parquet')
Xb,yb,db,cols_b = prep(E3)
Xn,yn,dn,cols_n = prep(NF)
Xs,ys,ds,cols_s = prep(SE)
base = ridge_eval(Xb,yb,db)[0]
greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
Xg = np.hstack([Xb]+[Xn[:,[cols_n.index(c)]] for c in greedy])
g_mae = ridge_eval(Xg,yb,db)[0]
print("base %.3f  greedy6 %.3f"%(base,g_mae))

# 1) day-drift terms
for name, v in [('day_idx', db.astype(float)), ('day_idx/100', db.astype(float)/100)]:
    print("  +%-12s %.3f (delta %+.3f)"%(name, ridge_eval(np.hstack([Xg,v[:,None]]),yb,db)[0], ridge_eval(np.hstack([Xg,v[:,None]]),yb,db)[0]-g_mae))

# 2) seasonal table features
for k,c in enumerate(cols_s):
    i = ridge_eval(np.hstack([Xg,Xs[:,[k]]]),yb,db)[0]
    print("  +%-18s %.3f (delta %+.3f)"%(c,i,i-g_mae))

# 3) per-snapshot standardized spend level: z-score spend_l123_mean within snapshot
E1 = api.load_saved('e001_txhist.parquet')
m = t.merge(E1[['household_key','snapshot_day','spend_l123_mean']], on=KEY, how='left')
v = m.spend_l123_mean.values.astype(float)
z = np.empty_like(v)
for s in np.unique(m.snapshot_day.values):
    idx = m.snapshot_day.values==s
    mu = np.nanmean(v[idx]); sd = np.nanstd(v[idx]) or 1
    z[idx] = (v[idx]-mu)/sd
print("  +zspend123      %.3f (delta %+.3f)"%(ridge_eval(np.hstack([Xg,z[:,None]]),yb,db)[0], ridge_eval(np.hstack([Xg,z[:,None]]),yb,db)[0]-g_mae))

# 4) rank-within-snapshot of spend level
r = np.empty_like(v)
for s in np.unique(m.snapshot_day.values):
    idx = m.snapshot_day.values==s
    r[idx] = pd.Series(v[idx]).rank(pct=True).values
print("  +rank_spend123  %.3f (delta %+.3f)"%(ridge_eval(np.hstack([Xg,r[:,None]]),yb,db)[0], ridge_eval(np.hstack([Xg,r[:,None]]),yb,db)[0]-g_mae))

# 5) full package test: greedy6 + hinges + zspend + rank
act = (t.merge(E1[['household_key','snapshot_day','zero_recent']], on=KEY, how='left').zero_recent==0).astype(float).values
s123 = m.spend_l123_mean.values.astype(float)
H = np.column_stack([s123*act, s123*(1-act), np.power(1+s123,0.9)*act, np.power(1+s123,0.9)*(1-act)])
Xfull = np.hstack([Xg, H, z[:,None], r[:,None]])
print("FULL package: %.3f (delta %+.3f)"%(ridge_eval(Xfull,yb,db)[0], ridge_eval(Xfull,yb,db)[0]-g_mae))

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

t = api.train_targets()
KEY=['household_key','snapshot_day']
def prep(df):
    m = t.merge(df, on=KEY, how='left')
    yv = m.future_spend_4w.values.astype(float); d = m.snapshot_day.values
    X = m.drop(columns=KEY+['future_spend_4w'])
    cols=[]
    for c in X.columns:
        s=X[c]
        if s.dtype==object or str(s.dtype).startswith('category') or s.dtype==bool:
            cols.append(pd.factorize(s)[0].astype(float))
        else:
            cols.append(pd.to_numeric(s,errors='coerce').values.astype(float))
    return np.column_stack(cols), yv, d, list(X.columns)
def ridge_eval(X,y,days,lams=(0.01,0.1,1,10,100),fitmax=347):
    fit=days<=fitmax; val=days>=375
    Xf=X[fit]; mu=np.nanmean(Xf,0); sd=np.nanstd(Xf,0); sd[sd<1e-9]=1
    Z=np.where(np.isfinite(X),(X-mu)/sd,0.0)
    Zf=np.hstack([Z[fit],np.ones((fit.sum(),1))]); Zv=np.hstack([Z[val],np.ones((val.sum(),1))])
    p=Zf.shape[1]; A=Zf.T@Zf; b=Zf.T@y[fit]; out=[]
    for lam in lams:
        w=np.linalg.solve(A+lam*np.eye(p),b); out.append((np.abs(Zv@w-y[val]).mean(),lam))
    out.sort(); return out[0]

E3 = api.load_saved('e003_catmix.parquet')
E1 = api.load_saved('e001_txhist.parquet')
NF = api.load_saved('nf_candidates.parquet')
X3,y3,d3,c3 = prep(E3)
X1,y1,d1,c1 = prep(E1)
Xn,yn,dn,cols_n = prep(NF)
greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
def add_greedy(X): return np.hstack([X]+[Xn[:,[cols_n.index(c)]] for c in greedy])
m = t.merge(E1[['household_key','snapshot_day','spend_l123_mean','zero_recent']], on=KEY, how='left')
s123 = m.spend_l123_mean.values.astype(float)
act = (m.zero_recent==0).astype(float).values
H = np.column_stack([s123*act, s123*(1-act), np.power(1+s123,0.9)*act, np.power(1+s123,0.9)*(1-act)])
day = d3.astype(float)[:,None]

P = {
 'P1 E003+g6+H+day': np.hstack([add_greedy(X3), H, day]),
 'P2 E001+g6+H+day': np.hstack([add_greedy(X1), H, day]),
 'P3 E003+g6+H':     np.hstack([add_greedy(X3), H]),
 'P4 E003+g6+day':   np.hstack([add_greedy(X3), day]),
 'P5 E003+g6+H+day (no pow90 hinge)': np.hstack([add_greedy(X3), H[:,[0,1]], day]),
}
for k,Xp in P.items():
    mae,lam = ridge_eval(Xp,y3,d3)
    print("%-38s %.3f (lam %g)"%(k,mae,lam))
# also lam grid finer for best
Xp = P['P1 E003+g6+H+day']
mae,lam = ridge_eval(Xp,y3,d3,lams=(1,3,10,30,100,300,1000))
print("P1 finer lam: %.3f (%g)"%(mae,lam))

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

E3 = api.load_saved('e003_catmix.parquet')
E1 = api.load_saved('e001_txhist.parquet')
NF = api.load_saved('nf_candidates.parquet')
greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
NFs = NF[['household_key','snapshot_day']+greedy]
E1s = E1[['household_key','snapshot_day','spend_l123_mean','zero_recent']]
df = E3.merge(NFs, on=['household_key','snapshot_day'], how='left').merge(E1s, on=['household_key','snapshot_day'], how='left')
s123 = df.spend_l123_mean.values.astype(float)
act = (df.zero_recent==0).astype(float).values
df['h_act_s123'] = s123*act
df['h_inact_s123'] = s123*(1-act)
df['h_act_pow90'] = np.power(1+s123,0.9)*act
df['h_inact_pow90'] = np.power(1+s123,0.9)*(1-act)
df['day_idx'] = df.snapshot_day.values.astype(float)
df = df.drop(columns=['spend_l123_mean','zero_recent'])
print(df.shape)
assert df[['household_key','snapshot_day']].duplicated().sum()==0
path = api.save_table(df, 'nf_p1.parquet')
print(path)
print(df.columns.tolist())

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

E3 = api.load_saved('e003_catmix.parquet')
E1 = api.load_saved('e001_txhist.parquet')
NF = api.load_saved('nf_candidates.parquet')
greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
NFs = NF[['household_key','snapshot_day']+greedy]
E1s = E1[['household_key','snapshot_day','zero_recent']]
df = E3.merge(NFs, on=['household_key','snapshot_day'], how='left').merge(E1s, on=['household_key','snapshot_day'], how='left')
s123 = df.spend_l123_mean.values.astype(float)
act = (df.zero_recent==0).astype(float).values
df['h_act_s123'] = s123*act
df['h_inact_s123'] = s123*(1-act)
df['h_act_pow90'] = np.power(1+s123,0.9)*act
df['h_inact_pow90'] = np.power(1+s123,0.9)*(1-act)
df['day_idx'] = df.snapshot_day.values.astype(float)
df = df.drop(columns=['zero_recent'])
print(df.shape)
assert df[['household_key','snapshot_day']].duplicated().sum()==0
path = api.save_table(df, 'nf_p1.parquet')
print(path)

# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

E3 = api.load_saved('e003_catmix.parquet')
NF = api.load_saved('nf_candidates.parquet')
greedy = ['nwmax12','nspend_l12','nspend7','nf_pow90_ewm4','nunits84','nf_nspend28_pow90']
NFs = NF[['household_key','snapshot_day']+greedy]
df = E3.merge(NFs, on=['household_key','snapshot_day'], how='left')
s123 = df.spend_l123_mean.values.astype(float)
act = (df.zero_recent==0).astype(float).values
df['h_act_s123'] = s123*act
df['h_inact_s123'] = s123*(1-act)
df['h_act_pow90'] = np.power(1+s123,0.9)*act
df['h_inact_pow90'] = np.power(1+s123,0.9)*(1-act)
df['day_idx'] = df.snapshot_day.values.astype(float)
print(df.shape)
assert df[['household_key','snapshot_day']].duplicated().sum()==0
assert df.spend_l123_mean.isna().sum()==0
path = api.save_table(df, 'nf_p1.parquet')
print(path)