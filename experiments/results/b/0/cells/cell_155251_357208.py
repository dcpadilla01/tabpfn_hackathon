import pandas as pd, numpy as np
v = agent_api.snapshot(459)
t = v.transactions; dm = v.display_mailer
print("dm dup combos:", int(dm.duplicated(['product_id','store_id','week_no']).sum()), "dm rows", len(dm))
dmap = {c:i for i,c in enumerate(['0','1','2','3','4','5','6','7','9','A'])}
mmap = {c:i for i,c in enumerate(['0','A','C','D','F','H','J','L','P','X','Z'])}
# aggregate txn to product-store-week
tg = t.groupby(['product_id','store_id','week_no'], as_index=False).agg(
    spend=('sales_value','sum'), qty=('quantity','sum'), nlines=('sales_value','size'),
    hh=('household_key','nunique'))
mg = tg.merge(dm, on=['product_id','store_id','week_no'], how='left')
mg['dv'] = mg.display.map(dmap).astype('float64')
mg['mv'] = mg.mailer.map(mmap).astype('float64')
mg['dv'] = mg.dv.fillna(0.0); mg['mv'] = mg.mv.fillna(0.0)
mg['disp'] = (mg.dv>0).astype(float); mg['mail'] = (mg.mv>0).astype(float)
print("matched share of combos:", round(mg.display.notna().mean(),4))
print("exposed spend share:", round(mg.loc[mg.disp>0,'spend'].sum()/mg.spend.sum(),4),
      "mailer:", round(mg.loc[mg.mail>0,'spend'].sum()/mg.spend.sum(),4))
print("dv>0 among matched:", round((mg.loc[mg.display.notna(),'dv']>0).mean(),4),
      "mv>0:", round((mg.loc[mg.display.notna(),'mv']>0).mean(),4))
# per-household join timing
hh = v.households
line = t[['household_key','product_id','store_id','week_no']].merge(
    mg[['product_id','store_id','week_no','spend','qty','nlines','dv','mv','disp','mail']],
    on=['product_id','store_id','week_no'], how='left').fillna(0.0)
g = line.groupby('household_key').agg(dm_spend=('spend','sum'), dm_qty=('qty','sum'),
      dm_trips=('nlines','sum'), dm_nprod=('product_id','nunique'),
      dm_disp_spend=('spend', lambda s: 0.0), dm_mail_spend=('spend','sum'))
# proper disp/mail sums via precomputed columns
line['dsp'] = line.spend*line.disp; line['msp'] = line.spend*line.mail
g2 = line.groupby('household_key').agg(dm_spend=('spend','sum'), dm_qty=('qty','sum'),
      dm_trips=('nlines','sum'), dm_nprod=('product_id','nunique'),
      dm_disp_spend=('dsp','sum'), dm_mail_spend=('msp','sum'))
print(g2.describe())
print("hh coverage:", g2.index.isin(hh).mean() if hasattr(g2.index,'isin') else 'n/a')
print("spend28 check: total spend all households", t.sales_value.sum(), "dm_spend sum", g2.dm_spend.sum())