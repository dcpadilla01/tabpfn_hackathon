import numpy as np, pandas as pd

try:
    df1 = load_saved('e001_txhist.parquet')
    print("E001 shape:", df1.shape)
    print("E001 cols:", list(df1.columns))
except Exception as e:
    print("ERR e001:", e)

v = snapshot()
try:
    print("view attrs:", [a for a in dir(v) if not a.startswith('_')])
    print("households:", type(v.households), "n=", len(v.households), "sample=", list(v.households)[:3])
    print("day,week:", v.day, v.week)
except Exception as e:
    print("ERR view:", e)

try:
    tx = v.transactions
    print("tx:", tx.shape, "days", tx.day.min(), tx.day.max())
    print(tx[['sales_value','coupon_disc','coupon_match_disc','retail_disc','trans_time']].describe().T)
except Exception as e:
    print("ERR tx:", e)

try:
    prod = v.products
    print("prod:", prod.shape)
    m = tx.merge(prod[['product_id','department','brand']], on='product_id', how='left')
    print("dept coverage:", round(float(m.department.notna().mean()), 4))
    dep = m.groupby('department').sales_value.sum().sort_values(ascending=False)
    print(dep.head(15))
    print("brand:", dict(prod.brand.value_counts(dropna=False).head(6)))
except Exception as e:
    print("ERR prod:", e)

try:
    tt = train_targets()
    print("targets:", tt.shape)
    print(tt.future_spend_4w.describe())
except Exception as e:
    print("ERR targets:", e)

try:
    K = 3.0
    def tfn(view, snapshot_day):
        hh = list(view.households)
        return pd.DataFrame({'k': [K]*len(hh)}, index=pd.Index(hh, name='household_key'))
    tf = build_features(tfn)
    print("bf test:", tf.shape, list(tf.columns)[:5])
    print(tf.groupby('snapshot_day').size().to_dict())
    print(tf.head(2))
except Exception as e:
    print("ERR bf:", e)
