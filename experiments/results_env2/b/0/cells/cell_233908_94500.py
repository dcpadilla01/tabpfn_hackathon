import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
import agent_api
def build(view, snapshot_day):
    tx = view.table('transactions')
    tx = tx[tx.sales_value.notna()].copy()
    tx['w'] = (tx.day + 8)//7
    week = (snapshot_day + 8)//7
    wk_tot = tx.groupby('w').sales_value.sum()
    wk_tot = wk_tot.reindex(range(1, week+1)).ffill()
    lvl = wk_tot / wk_tot.iloc[:8].mean()
    g = tx.groupby(['household_key','w']).sales_value.sum()
    weeks = list(range(max(1, week-25), week+1))
    idx = pd.MultiIndex.from_product([view.households, weeks], names=['household_key','w'])
    mat = g.reindex(idx).fillna(0.0).unstack()
    mat = mat.reindex(view.households)
    if mat.shape[1] < 26:
        pad = pd.DataFrame(np.nan, index=mat.index, columns=range(1, 27-mat.shape[1]))
        mat = pd.concat([pad, mat], axis=1)
    f = pd.DataFrame(index=view.households)
    f['mkt_lvl'] = float(lvl.loc[weeks].mean())
    f['mkt_lvl_last4'] = float(lvl.loc[weeks[-4:]].mean())
    f['mkt_hol_ahead'] = float((week%52==0) or ((week+1)%52==1))
    f['mkt_hol_now'] = float(week%52==0)
    # holiday weights aligned to the 26 padded columns (w = week-25 .. week; 0 if w<1)
    warr = np.array([w if w>=1 else 0 for w in range(week-25, week+1)])
    wmod = np.where(warr>0, ((warr-1)%52)+1, 0)
    hol = ((wmod>=51)|(wmod==1)).astype(float)
    V = np.nan_to_num(mat.values)
    f['hh_hol_wk_share'] = (V*hol).sum(1)/np.maximum(1e-9, V.sum(1))
    f['hh_hol_wk_amt'] = (V*hol).sum(1)
    r = mat.rolling(4, axis=1, min_periods=4).mean().values*4
    f['r4_lag1'] = r[:,-1]; f['r4_lag2'] = r[:,-2]; f['r4_lag3'] = r[:,-3]
    f['r4_lag4'] = r[:,-4]; f['r4_lag5'] = r[:,-5]
    f['r4_lag13'] = r[:,-13]; f['r4_lag14'] = r[:,-14]; f['r4_lag26'] = r[:,-26]
    with np.errstate(all='ignore'):
        f['r4_ratio_l2_l14'] = r[:,-2]/(r[:,-14]+1)
        f['r4_ratio_l1_l13'] = r[:,-1]/(r[:,-13]+1)
        f['r4_ratio_l13_l26'] = r[:,-13]/(r[:,-26]+1)
    f['r4_mean_l1_l26'] = np.nanmean(r, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f['wk_max'] = np.nanmax(mat.values, axis=1)
        f['wk_std'] = np.nanstd(mat.values, axis=1)
    f['wk_last'] = mat.values[:,-1]
    f['wk_active'] = (np.nan_to_num(mat.values)>0).sum(1)
    d0 = snapshot_day-83
    txd = tx[(tx.day>=d0)&(tx.day<=snapshot_day)]
    agg = txd.groupby('household_key').agg(sales=('sales_value','sum'),
        coup=('coupon_disc','sum'), match=('coupon_match_disc','sum'), retail=('retail_disc','sum'),
        nlines=('sales_value','size'))
    agg = agg.reindex(view.households).fillna(0.0)
    f['disc_retail_84'] = -agg.retail
    f['disc_coup_84'] = -agg.coup
    f['disc_match_84'] = -agg.match
    f['disc_total_84'] = -(agg.retail+agg.coup+agg.match)
    f['disc_share_84'] = (-(agg.retail+agg.coup+agg.match))/(agg.sales+1)
    f['lines_84'] = agg.nlines
    f['lines_per_dollar_84'] = agg.nlines/(agg.sales+1)
    prod = view.table('products')
    dept = prod.set_index('product_id').department
    txd = txd.assign(dept=txd.product_id.map(dept))
    gd = txd.groupby(['household_key','dept']).sales_value.sum().unstack().reindex(view.households).fillna(0.0)
    tot = gd.sum(1)
    for c in ['GROCERY','PRODUCE','MEAT','MEAT-PCKGD','DRUG GM','DELI','PASTRY','KIOSK-GAS','NUTRITION','SEAFOOD-PCKGD','SALAD BAR','COSMETICS','MISC SALES TRAN','FLORAL','SPIRITS','SEAFOOD']:
        f['d84_'+c] = gd[c]/(tot+1) if c in gd.columns else 0.0
    f['d84_n_depts'] = (gd>0).sum(1)
    p = (gd.values/(tot.values[:,None]+1e-9))
    f['d84_entropy'] = -(np.where(gd.values>0, p, 0)*np.log(p.clip(1e-9))).sum(1)
    return f

t0=time.time()
bf = agent_api.build_features(build)
print('build time', round(time.time()-t0,1), bf.shape)
tt = agent_api.train_targets()
m = tt.merge(bf, on=['household_key','snapshot_day'])
print('merged', m.shape)
res=[]
for c in bf.columns:
    x=m[c]; ok=x.notna()
    res.append((c, np.corrcoef(x[ok], m.future_spend_4w[ok])[0,1], ok.mean()))
res=pd.DataFrame(res, columns=['feat','corr','n']).sort_values('corr', key=lambda s: s.abs(), ascending=False)
print(res.head(28).to_string())
print(res.tail(12).to_string())
path = agent_api.save_table(bf, 'e014_seasonal.parquet')
print('saved', path)
