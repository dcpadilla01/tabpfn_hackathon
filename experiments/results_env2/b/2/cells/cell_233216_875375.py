tt = train_targets()
y = tt.future_spend_4w
print('n', len(tt), 'mean %.1f median %.1f zero%% %.3f' % (y.mean(), y.median(), (y==0).mean()))
print('quantiles', y.quantile([.5,.75,.9,.95,.99]).to_dict())
v = snapshot()
tr = v.transactions
print('transactions up to 459:', tr.shape)
tr2 = tr.assign(dow=tr.day % 7)
print(tr2.groupby('dow').sales_value.agg(['mean','count']))
print('trans_time quantiles', tr.trans_time.quantile([.1,.5,.9]).to_dict())
d = tr[['sales_value','coupon_disc','coupon_match_disc','retail_disc','quantity']]
print(d.describe().loc[['mean','50%','max']])
