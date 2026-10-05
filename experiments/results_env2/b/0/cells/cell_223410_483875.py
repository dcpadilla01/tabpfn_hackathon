import agent_api as A
import pandas as pd, numpy as np
snap = A.snapshot()
tx = snap.transactions
# store diversity: how many stores per household in trailing 84d
g = tx[tx.day > 459-84].groupby('household_key')
stores_hh = g.store_id.nunique()
print('stores per hh (84d):', stores_hh.describe())
print('top_store_share_84 already in E006')
# check quantity vs sales correlation, and big-quantity rows
print('corr qty-sales:', tx.quantity.corr(tx.sales_value))
# zero/negative sales lines
print('zero sales lines:', (tx.sales_value<=0).mean())
# baskets per household 84d
b = tx[tx.day>375].groupby('household_key').basket_id.nunique()
print('baskets 84d:', b.describe())
# check E006 target relationship quickly
e6 = A.load_saved('e006_zero_inflation.parquet')
print(e6.shape, e6.snapshot_day.unique())
