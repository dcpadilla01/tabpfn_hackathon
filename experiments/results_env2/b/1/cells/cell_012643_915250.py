
import pandas as pd, numpy as np
v = agent_api.snapshot(459)
tx = v.transactions
demo = v.demographics

# Household-level aggregates for brand mix and demographics
g = tx.groupby('household_key')
priv = g.apply(lambda d: (d.sales_value[d.brand_of_product[d.product_id].values=='Private'].sum()/max(d.sales_value.sum(),1e-9)) if False else np.nan)
