import warnings; warnings.filterwarnings('ignore')
v = snapshot()
tx = v.transactions
hh = list(v.households)
print('n hh', len(hh))
g = tx.groupby('household_key').sales_value.sum()
print('groupby ok', len(g))
dm = v.display_mailer
dm_small = dm[['product_id','store_id','week_no','display','mailer']]
last28 = tx[tx.day >= 459-27]
j = last28.merge(dm_small, on=['product_id','store_id','week_no'], how='left')
print('merge ok', j.shape, 'match rate', j.display.notna().mean().round(3))
print('ct', v.campaign_targets.shape, 'cr', v.coupon_redemptions.shape, 'camps', v.campaigns.shape)
print(v.campaign_targets.description.value_counts())
print(v.demographics.shape)