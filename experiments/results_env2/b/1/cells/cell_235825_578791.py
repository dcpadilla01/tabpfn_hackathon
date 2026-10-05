
import agent_api as A, pandas as pd, numpy as np, time
t0=time.time()
v = A.snapshot(459)
tx = v.table('transactions')
print('tx rows', len(tx))
dm = v.table('display_mailer')
promo = dm[(dm.display>0)|(dm.mailer!='0')][['product_id','store_id','week_no']]
print('promo rows total', len(promo), 'time', round(time.time()-t0,1))
# commodity-level promo intensity: product->commodity map
p = v.table('products')[['product_id','commodity_desc']]
promo_c = promo.merge(p, on='product_id', how='left')
ci = promo_c.groupby(['commodity_desc','week_no']).size().rename('n_promo').reset_index()
print('commodity-week promo table', ci.shape, 'time', round(time.time()-t0,1))
# household recent spend by commodity for a test snapshot (431)
s=431; wk=(s+8)//7
t = tx[tx.day<=s].copy(); t=t[t.day>s-84]; t['week_no']=(t['day']+8)//7
t=t[t.week_no<=wk]
tc = t.merge(p, on='product_id', how='left').groupby('household_key').apply(
    lambda g: g.merge(ci[ci.week_no.between(wk-12,wk)], on=['commodity_desc','week_no'], how='left')['n_promo'].fillna(0).sum()/max(len(g),1))
print('sample exposure computed', tc.shape, 'time', round(time.time()-t0,1))
print(tc.describe())
