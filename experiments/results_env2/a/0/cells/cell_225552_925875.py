import pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
prod = agent_api.snapshot(459).products
print(prod.department.value_counts().head(30))
print(prod.brand.value_counts())
