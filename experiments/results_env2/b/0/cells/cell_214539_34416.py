import agent_api
import pandas as pd, numpy as np

e5 = agent_api.load_saved('e005_trend_season.parquet')
print('e5 shape', e5.shape)
print('e5 cols:', list(e5.columns))

tt = agent_api.train_targets()
print('targets shape', tt.shape)
print(tt['future_spend_4w'].describe())

print('snapshot days:', agent_api.snapshot_days())

v = agent_api.snapshot(459)
tx = v.transactions
print(tx[['sales_value','quantity','retail_disc','coupon_disc','coupon_match_disc','trans_time','day']].describe())
print(tx.head(3))
