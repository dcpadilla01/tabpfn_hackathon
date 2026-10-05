import pandas as pd, numpy as np, time
from agent_api import snapshot, train_targets, load_saved, KEYS, TARGET

s459 = snapshot()
prod = s459.products
print('products', prod.shape)
print(prod.department.value_counts().head(15))
print(prod.brand.value_counts())
camp = s459.campaigns; print('campaigns', camp.shape, camp.description.value_counts().to_dict())
print(camp.head(3))
ctarg = s459.campaign_targets; print('ctarg', ctarg.shape, ctarg.description.value_counts().to_dict())
cred = s459.coupon_redemptions; print('cred', cred.shape)
