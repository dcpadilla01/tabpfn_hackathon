import pandas as pd, numpy as np
try:
    _a = agent_api
except NameError:
    import agent_api as _a

t = _a.load_saved('e003_catmix.parquet')
print('e003 shape', t.shape)
print('e003 cols:', list(t.columns))

tt = _a.train_targets()
y = tt['future_spend_4w']
print('\ntargets', tt.shape)
print(y.describe())
print('zero share', float((y==0).mean()))
print(y.quantile([.05,.1,.25,.5,.75,.9,.95,.99]).to_dict())

v = _a.snapshot()
print('\nview day/week:', v.day, v.week)
print('households type', type(v.households), 'len', len(v.households))
tx = v.transactions
print('tx shape', tx.shape)
print(tx[['sales_value','quantity','retail_disc','coupon_disc','coupon_match_disc']].describe())
print('neg sales share', float((tx.sales_value<0).mean()), 'qty<=0 share', float((tx.quantity<=0).mean()))
print('hh with tx', tx.household_key.nunique())