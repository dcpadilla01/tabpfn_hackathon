import agent_api, pandas as pd, numpy as np

def probe(view, day):
    hh = view.households
    tx = view.transactions
    g = tx.groupby('household_key').sales_value.sum()
    return pd.DataFrame({'spend_probe': g.reindex(hh).values}, index=hh)

tab = agent_api.build_features(probe)
print('build_features shape', tab.shape)

v = agent_api.snapshot()
dm = v.display_mailer
print(dm.display.value_counts(dropna=False).to_dict())
print(dm.mailer.value_counts(dropna=False).to_dict())
print(dm.head(3))
print('weeks range:', dm.week_no.min(), dm.week_no.max())
tx=v.transactions
print('tx days:', tx.day.min(), tx.day.max())
print('coupon_redemptions:'); print(v.coupon_redemptions.head(3))
print('campaign_targets:'); print(v.campaign_targets.head(3))
print('campaigns:'); print(v.campaigns)