import agent_api, pandas as pd, numpy as np

def probe(view, day):
    hh = view.households
    print(f"day={day} week={view.week} n_hh={len(hh)}",
          "tx:", view.transactions.shape,
          "camp:", view.campaigns.shape,
          "ct:", view.campaign_targets.shape,
          "red:", view.coupon_redemptions.shape,
          "dm:", view.display_mailer.shape,
          "prods:", view.products.shape)
    print("camp desc:", view.campaigns.description.value_counts().to_dict())
    print("ct desc:", view.campaign_targets.description.value_counts().to_dict())
    print("ct hh:", view.campaign_targets.household_key.nunique())
    return pd.DataFrame({'x': 1.0}, index=pd.Index(hh, name='household_key'))

bf = agent_api.build_features(probe)
print(bf.shape)
print(bf.head())