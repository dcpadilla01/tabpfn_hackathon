v = snapshot()
cr = v.coupon_redemptions
print('coupon_redemptions', cr.shape, cr.columns.tolist())
print(cr.head())
print('n households redeeming:', cr.household_key.nunique())
print('redemptions per day quantiles:', cr.groupby('day').size().quantile([.5,.9]).to_dict())
tr = v.transactions
# promo dependence quick check
import numpy as np
tr['promo'] = (tr.coupon_disc>0)|(tr.coupon_match_disc>0)|(tr.retail_disc>0)
print('share of line items with any promo:', tr.promo.mean())
print('share of spend with promo:', (tr.sales_value[tr.promo]).sum()/tr.sales_value.sum())
