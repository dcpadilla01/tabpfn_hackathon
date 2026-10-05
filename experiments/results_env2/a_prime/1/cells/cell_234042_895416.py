import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
T = agent_api.load_saved('e012_style.parquet')
tt = agent_api.train_targets()
df = T.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
X = df[num].astype(float).copy()
X = X.replace([np.inf,-np.inf], np.nan)
print("NaN per col max:", X.isna().sum().max())
print("cols all-NaN:", [c for c in num if X[c].isna().all()])
X = X.fillna(X.median())
print("any NaN left:", np.isnan(X.values).any(), "any inf:", np.isinf(X.values).any())
print("y NaN:", np.isnan(y).sum())
print("y max:", y.max())
A = X.T@X
print("A any nan:", np.isnan(A).any(), "A max:", np.nanmax(A))
