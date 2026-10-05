import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
# Precompute static-ish lookups for speed: dept map, and check timing of groupby ops
s = agent_api.snapshot()
tx = s.table('transactions')
prod = s.table('products')
dept = prod.set_index('product_id').department
t0=time.time() if False else None
import time as _t
t0=_t.time()
tx['w']=(tx.day+8)//7
g = tx.groupby(['household_key','w']).sales_value.sum()
print('hh-week groupby time', round(_t.time()-t0,2))
t0=_t.time()
tx['dept'] = tx.product_id.map(dept)
g2 = tx.groupby(['household_key','dept']).sales_value.sum()
print('hh-dept groupby time', round(_t.time()-t0,2), g2.shape)
t0=_t.time()
g3 = tx.groupby(['household_key','w','dept']).sales_value.sum()
print('hh-week-dept time', round(_t.time()-t0,2), g3.shape)
