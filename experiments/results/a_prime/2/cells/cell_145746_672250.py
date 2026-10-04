import agent_api as api, pandas as pd, numpy as np
for n in ['recency_agg.parquet','dept_mix_recency.parquet','temporal_structure.parquet','basket_tenure.parquet','marketing_exposure.parquet','behavioral_candidates.parquet']:
    df = api.load_saved(n)
    print(n, df.shape)
bt = api.load_saved('basket_tenure.parquet'); print('BT cols:', list(bt.columns))
bc = api.load_saved('behavioral_candidates.parquet'); print('BC cols:', list(bc.columns))
v = api.snapshot()
t = v.transactions
print('tx', t.shape)
print(t[['sales_value','quantity','retail_disc','coupon_disc','coupon_match_disc']].describe().T)
p = v.products
print('prod', p.shape)
print(p['brand'].value_counts(dropna=False).head())
print('dept nunique', p['department'].nunique(), 'commod nunique', p['commodity_desc'].nunique())
tt = api.train_targets()
print('targets', tt.shape); print(tt['future_spend_4w'].describe())
print('households type', type(v.households), len(v.households))
print('cr', v.coupon_redemptions.shape)