import agent_api as A
snap = A.snapshot()
tx = snap.transactions
print('stores:', tx.store_id.nunique(), 'products:', tx.product_id.nunique(), 'hh:', tx.household_key.nunique())
print(tx[['sales_value','quantity']].describe())
print('coupon cols sum:', tx[['coupon_match_disc','coupon_disc','retail_disc']].sum())
# campaign types
print(snap.campaign_targets.description.value_counts())
print(snap.campaigns.head())
print('n campaigns started by 459:', len(snap.campaigns))
