
import agent_api as A, pandas as pd, numpy as np, time
t0=time.time()
v = A.snapshot(459)
tx = v.table('transactions')
dm = v.table('display_mailer')
dm['disp'] = dm['display'].astype(str).astype(float)
dm['mail'] = dm['mailer'].astype(str)
promo = dm[(dm.disp>0)|(dm.mail!='0')][['product_id','store_id','week_no']]
print('promo rows', len(promo), 'time', round(time.time()-t0,1))
p = v.table('products')[['product_id','commodity_desc']]
promo_c = promo.merge(p, on='product_id', how='left')
ci = promo_c.groupby(['commodity_desc','week_no']).size().rename('n_promo').reset_index()
print('commodity-week promo', ci.shape, 'time', round(time.time()-t0,1))
# household exposure at snapshot 431: share of recent spend in promo-active commodity-weeks
s=431; wk=(s+8)//7
t = tx[(tx.day<=s)&(tx.day>s-84)].copy()
t['week_no']=(t['day']+8)//7
tc = t.merge(p, on='product_id', how='left')
tc = tc.merge(ci[ci.week_no.between(wk-12,wk)], on=['commodity_desc','week_no'], how='left')
tc['promo'] = tc['n_promo'].notna().astype(float)
hh = tc.groupby('household_key').agg(spend=('sales_value','sum'), promo_spend=('sales_value', lambda x: 0))
# simpler: promo_spend = sum of sales where promo==1
g = tc.groupby(['household_key','promo'])['sales_value'].sum().unstack(fill_value=0)
g.columns = ['sp0','sp1'] if 0 in g.columns and 1 in g.columns else list(g.columns)
exp = (g.get(1,0)/ (g.get(0,0)+g.get(1,0)+1e-9)).rename('promo_share')
print('exposure sample', exp.shape, 'time', round(time.time()-t0,1))
print(exp.describe())
