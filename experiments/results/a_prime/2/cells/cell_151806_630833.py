import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
v = agent_api.snapshot()
print('campaigns:', v.campaigns.shape); print(v.campaigns.head(3))
print('desc values:', v.campaigns['description'].value_counts().to_string())
print('campaign_targets:', v.campaign_targets.shape); print(v.campaign_targets.head(3))
print('coupon_redemptions:', v.coupon_redemptions.shape)
dm = v.display_mailer
print('display_mailer:', dm.shape)
print(dm.head(3))
print('display vals:', dm['display'].value_counts().head(8).to_string())
print('mailer vals:', dm['mailer'].value_counts().head(8).to_string())
tx = v.transactions
print('tx up to 459:', tx.shape, 'max day', tx['day'].max())
prod = v.products[['product_id','department']].drop_duplicates('product_id')
ds = tx.merge(prod, on='product_id', how='left').groupby('department')['sales_value'].sum().sort_values(ascending=False)
print('top depts:', list(ds.head(16).index))