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