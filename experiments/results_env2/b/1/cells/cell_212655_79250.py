
import pandas as pd, numpy as np
import agent_api as A

print(A.snapshot_days())
s = A.snapshot()
print('view:', type(s), 'day', s.day, 'week', s.week)
tx = s.transactions
print('tx shape', tx.shape)
print(tx.head(3))
print(tx[['sales_value','quantity','coupon_disc','retail_disc','coupon_match_disc']].describe())
print('households in tx:', tx.household_key.nunique())
hh = s.households
print('households type', type(hh), 'len', len(hh))
print(list(hh[:5]) if not isinstance(hh, pd.DataFrame) else hh.head())
h = A.history(tx.household_key.iloc[0])
print('history shape', h.shape)
print(h.head(3))
d = s.demographics
print('demographics', d.shape)
ct = s.campaign_targets
print('campaign_targets', ct.shape)
print(ct.description.value_counts().head())
cr = s.coupon_redemptions
print('coupon_redemptions', cr.shape)

# quick predictive check: past 28d spend vs target at snapshot 431
tt = A.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
s431 = A.snapshot(431)
t431 = s431.transactions
sub = tt[tt.snapshot_day == 431].set_index('household_key')
t = t431[t431.day > 431-28]
sp28 = t.groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
t84 = t431[t431.day > 431-84]
sp84 = t84.groupby('household_key').sales_value.sum().reindex(sub.index).fillna(0)
print('corr sp28 vs target', np.corrcoef(sp28, sub.future_spend_4w)[0,1])
print('corr sp84 vs target', np.corrcoef(sp84, sub.future_spend_4w)[0,1])
print('MAE of predicting target=sp28:', (sp28 - sub.future_spend_4w).abs().mean())
print('MAE of predicting target=sp84/3:', (sp84/3 - sub.future_spend_4w).abs().mean())
print('MAE of predicting target=mean:', (sub.future_spend_4w.mean() - sub.future_spend_4w).abs().mean())
