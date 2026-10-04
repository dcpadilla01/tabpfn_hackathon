import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
m = A.load_saved('e013_stock.parquet')
print(m.dtypes.value_counts())
print('m household dtype', m.household_key.dtype, m.household_key.head(3).tolist())
v=A.snapshot(459); print('tx household dtype', v.transactions.household_key.dtype, v.transactions.household_key.head(3).tolist())
