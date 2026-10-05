import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
t = agent_api.snapshot(459).transactions
prod = agent_api.snapshot(459).products
m = t.merge(prod[['product_id','commodity_desc']], on='product_id', how='left')
# repeat-purchase structure: fraction of line items that are a re-buy of same commodity in prior 28d
# quick proxy: per household, share of spend in top-3 commodities, and commodity loyalty (repeat rate)
h = m.groupby('household_key')
# commodity concentration (HHI) of spend over trailing 84d
t84 = m[m.day > 459-84]
sp = t84.groupby(['household_key','commodity_desc']).sales_value.sum().reset_index()
tot = sp.groupby('household_key').sales_value.transform('sum')
sp['share'] = sp.sales_value/tot
hhi = (sp.assign(s2=sp.share**2).groupby('household_key').s2.sum())
print('HHI describe:', hhi.describe().round(3).to_dict())
# repeat rate: for each household, fraction of spend in trailing 84d on commodities bought in previous 84d window (84-168)
t168 = m[(m.day > 459-168) & (m.day <= 459-84)]
prev_comm = set(zip(t168.household_key, t168.commodity_desc))
sp['is_repeat'] = [ (h,c) in prev_comm for h,c in zip(sp.household_key, sp.commodity_desc)]
rep = sp.groupby('household_key').apply(lambda d: np.average(d.is_repeat, weights=d.sales_value))
print('repeat-rate describe:', rep.describe().round(3).to_dict())
# stock-up indicator: basket size spikes
b = t.groupby(['household_key','basket_id']).agg(sales=('sales_value','sum'), day=('day','first'))
b = b.sort_values(['household_key','day'])
b['prev'] = b.groupby('household_key').sales.shift(1)
b['ratio'] = b.sales/b.prev
print('basket sales ratio>2.5 frac:', (b.ratio>2.5).mean().round(3))
print('time to check per-household ops feasible')
