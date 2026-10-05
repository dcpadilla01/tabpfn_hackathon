
import agent_api as A, numpy as np, pandas as pd, time
t0=time.time()
v = A.snapshot(459)
tx = v.table('transactions')[['household_key','day','basket_id','product_id','quantity','sales_value','store_id','week_no']].copy()
dm = v.table('display_mailer')
def bmask(s):
    sn = pd.to_numeric(s, errors='coerce')
    if sn.notna().all(): return (sn>0).values
    return (~s.astype(str).str.strip().isin(['0','0.0','','nan','None'])).values
dmk = dm[['product_id','store_id','week_no']].copy()
dmk['d'] = bmask(dm.display); dmk['m'] = bmask(dm.mailer)
dm_d = dmk[dmk.d][['product_id','store_id','week_no']].drop_duplicates(); dm_d['ed']=1.0
dm_m = dmk[dmk.m][['product_id','store_id','week_no']].drop_duplicates(); dm_m['em']=1.0
prod = v.table('products')[['product_id','commodity_desc']]
feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
train_days = sorted(tt.snapshot_day.unique())
out = []
for d in train_days:
    w = tx[tx.day<=d]
    w84 = w[w.day>d-84]; w28 = w[w.day>d-28]; w7 = w[w.day>d-7]
    g = w84.groupby('household_key')
    f = pd.DataFrame(index=g.size().index)
    f['units_84'] = g.quantity.sum()
    f['spend84'] = g.sales_value.sum()
    f['trips_7'] = w7.groupby('household_key').basket_id.nunique()
    f['nprod_28'] = w28.groupby('household_key').product_id.nunique()
    w28c = w28.merge(prod, on='product_id', how='left')
    f['ncomm_28'] = w28c.groupby('household_key').commodity_desc.nunique()
    q = w84.quantity.where(w84.quantity>0)
    f['up_mean_84'] = (w84.sales_value/q).groupby(w84.household_key).mean()
    glob_up = (w84.sales_value/q).mean()
    f['premium_idx'] = f.up_mean_84/glob_up
    st = w84.groupby('store_id').agg(sv=('sales_value','sum'), nh=('household_key','nunique'))
    st['rich'] = st.sv/st.nh
    ss = w84.groupby(['household_key','store_id'], as_index=False).sales_value.sum()
    top = ss.sort_values('sales_value').groupby('household_key').tail(1).set_index('household_key')
    f['store_rich_84'] = st.rich.reindex(top.store_id).values
    j = w84.merge(dm_d, on=['product_id','store_id','week_no'], how='left')
    j['ed'] = j.ed.fillna(0.0); j['sve'] = j.sales_value*j.ed
    den = j.groupby('household_key').sales_value.sum()
    f['disp_share_84'] = j.groupby('household_key').sve.sum()/den
    j2 = w84.merge(dm_m, on=['product_id','store_id','week_no'], how='left')
    j2['em'] = j2.em.fillna(0.0); j2['svm'] = j2.sales_value*j2.em
    f['mail_share_84'] = j2.groupby('household_key').svm.sum()/den
    f['snapshot_day'] = d
    out.append(f.reset_index().rename(columns={'index':'household_key'}))
cand = pd.concat(out, ignore_index=True)
print('cand', cand.shape, 'elapsed %.0fs' % (time.time()-t0))
oof = A.load_saved('oof_e013.parquet')
m = tt.merge(oof, on=['household_key','snapshot_day']).merge(cand, on=['household_key','snapshot_day'])
m['resid'] = m.future_spend_4w - m.oof
m['r_7_28'] = m.spend_7/(m.spend_28+1); m['r_28_84'] = m.spend_28/(m.spend_84+1); m['r_84_168'] = m.spend_84/(m.spend_168+1)
m['lag_std'] = m[['lag1_spend','lag2_spend','lag3_spend']].std(axis=1)
W = m[['wk%d'%i for i in range(8)]].values; t8 = np.arange(8)
m['wk_slope'] = ((W*t8).sum(1)*8 - W.sum(1)*t8.sum())/(8*(t8**2).sum()-t8.sum()**2)
cc = [c for c in m.columns if c not in ('household_key','snapshot_day','oof','y','future_spend_4w','resid')]
for c in cc:
    x = m[c].fillna(m[c].median())
    print('%-16s corr_resid %+.4f  corr_y %+.4f' % (c, np.corrcoef(x, m.resid)[0,1], np.corrcoef(x, m.future_spend_4w)[0,1]))
A.save_table(cand, 'cand_train.parquet')
print('saved. elapsed %.0fs' % (time.time()-t0))
