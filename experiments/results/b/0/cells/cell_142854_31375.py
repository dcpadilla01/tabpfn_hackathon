import agent_api as api
import pandas as pd, numpy as np

def probe(view, day):
    print("DAY", day, "week", view.day, view.week)
    hh = view.households
    print("n hh:", len(hh), type(hh))
    print(hh.head() if hasattr(hh,'head') else hh[:5])
    tx = view.transactions
    print("tx shape:", tx.shape, "max day:", tx['day'].max())
    print(tx.head(2))
    print("demo:", view.demographics.shape)
    print("campaign_targets:", view.campaign_targets.shape)
    print("coupons:", view.coupons.shape)
    print("coupon_redemptions:", view.coupon_redemptions.shape)
    print("display_mailer:", view.display_mailer.shape)
    print("campaigns:", view.campaigns.shape)
    print("products:", view.products.shape)
    return pd.DataFrame(index=hh[:3])

df = api.build_features(probe)
print(df.shape)
print(df.head())
