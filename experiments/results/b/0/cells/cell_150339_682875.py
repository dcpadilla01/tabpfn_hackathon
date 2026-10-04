import warnings; warnings.filterwarnings('ignore')
v = snapshot()
tx = v.transactions
dm = v.display_mailer
dm_small = dm[['product_id','store_id','week_no','display','mailer']]
last28 = tx[tx.day >= 459-27]
j = last28.merge(dm_small, on=['product_id','store_id','week_no'], how='left')
print('merge ok', j.shape, 'match rate', j.display.notna().mean().round(3))
print('n hh in last28', j.household_key.nunique())
print('ct', v.campaign_targets.shape, 'cr', v.coupon_redemptions.shape, 'camps', v.campaigns.shape)
print(v.campaign_targets.description.value_counts())
print('demo', v.demographics.shape)
print('hh via tx', tx.household_key.nunique())