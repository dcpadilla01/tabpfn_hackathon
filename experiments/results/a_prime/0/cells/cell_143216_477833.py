import pandas as pd, numpy as np
import agent_api as api

v2 = api.load_saved('hist_v2.parquet')
print('v2 shape', v2.shape)
print('v2 cols', list(v2.columns))

v0 = api.snapshot()
tx = v0.transactions
tp = tx.merge(v0.products[['product_id','department']], on='product_id', how='left')
dep = tp.groupby('department')['sales_value'].sum().sort_values(ascending=False)
print('\nn departments:', tp.department.nunique())
print(dep.head(22))

print('\ncampaigns:', v0.campaigns.shape)
print(v0.campaigns.head(3))
print(v0.campaigns.description.value_counts())

cr = v0.coupon_redemptions
print('\nredemptions:', cr.shape)
print(cr.head(3))

dm = v0.display_mailer
print('\ndisplay_mailer:', dm.shape)
print('display vals:', dm.display.value_counts().head(8).to_dict())
print('mailer vals:', dm.mailer.value_counts().head(8).to_dict())

print('\ntx shape', tx.shape, 'day range', tx.day.min(), tx.day.max())
