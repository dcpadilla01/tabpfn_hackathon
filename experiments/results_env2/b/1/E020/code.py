
import pandas as pd, numpy as np
base = agent_api.load_saved('e008_level_shape.parquet')
print("base shape:", base.shape)
print(sorted([c for c in base.columns if c not in ('household_key','snapshot_day')]))
v = agent_api.snapshot(459)
print("\nbrand values:\n", v.products.brand.value_counts(dropna=False).head())
print("\nclassification_4 values:\n", v.demographics.classification_4.value_counts(dropna=False))
tx = v.transactions
print("\ntx shape:", tx.shape, "dtypes:\n", tx.dtypes)
print(tx[['trans_time','quantity','sales_value']].head(3))


# ---- cell ----

import pandas as pd, numpy as np
v = agent_api.snapshot(459)
tx = v.transactions
demo = v.demographics

# Household-level aggregates for brand mix and demographics
g = tx.groupby('household_key')
priv = g.apply(lambda d: (d.sales_value[d.brand_of_product[d.product_id].values=='Private'].sum()/max(d.sales_value.sum(),1e-9)) if False else np.nan)
