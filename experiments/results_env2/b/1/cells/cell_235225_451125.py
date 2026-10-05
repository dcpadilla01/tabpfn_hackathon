
import agent_api as A, pandas as pd, numpy as np
for name in ['e008_level_shape','e006_cadence','e007_temporal','e001_history','e003_full']:
    df = A.load_saved(name + '.parquet')
    print('==', name, df.shape)
    print(list(df.columns))
tt = A.train_targets()
print('targets', tt.shape)
print(tt['future_spend_4w'].describe())
v = A.snapshot()
tx = v.table('transactions')
print(tx.dtypes)
print('neg sales', int((tx.sales_value<0).sum()), 'qty<=0', int((tx.quantity<=0).sum()))
p = v.table('products')
print(p['brand'].value_counts(dropna=False).head(10))
print('n commodities', p['commodity_desc'].nunique(), 'n depts', p['department'].nunique())
print('households type:', type(v.households))
print(list(pd.Index(v.households))[:3] if not isinstance(v.households, pd.DataFrame) else v.households.head(3))
