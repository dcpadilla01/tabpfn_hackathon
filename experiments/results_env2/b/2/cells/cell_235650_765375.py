import agent_api, numpy as np, pandas as pd

tx0 = agent_api.snapshot().table('transactions')
print(tx0[['coupon_disc','coupon_match_disc','retail_disc','sales_value']].describe().round(2))
dm0 = agent_api.snapshot().table('display_mailer')
print(dm0.shape); print(dm0.head(3))
print('display vals', dm0['display'].value_counts(dropna=False).head(8).to_dict())
print('mailer vals', dm0['mailer'].value_counts(dropna=False).head(8).to_dict())
