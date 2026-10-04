import agent_api as A, pandas as pd, numpy as np
for name in ['e009_demo','mkt_v2','hist_v2','structure_v1','season_v1','mix_v1','e006_temporal']:
    df = A.load_saved(name+'.parquet')
    print('==', name, df.shape)
    print(list(df.columns))
v = A.snapshot()
dm = v.display_mailer
print('\nDM shape', dm.shape)
print(dm.head(3).to_string())
print('display:', dm.display.value_counts(dropna=False).head(12).to_dict())
print('mailer:', dm.mailer.value_counts(dropna=False).head(12).to_dict())
tx = v.transactions
print('\nTX describe:')
print(tx[['sales_value','quantity','coupon_disc','retail_disc','coupon_match_disc','trans_time']].describe().to_string())
print('households type:', type(v.households))
print(A.KEYS, A.TARGET)
