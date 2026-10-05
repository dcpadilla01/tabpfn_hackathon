
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
A.save_table(cand, 'cand_train.parquet')
print('cand', cand.shape, 'saved OK. elapsed %.0fs' % (time.time()-t0))
