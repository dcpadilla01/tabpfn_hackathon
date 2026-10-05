
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
