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


# ---- cell ----
import numpy as np, pandas as pd

DEPTS = ['GROCERY','DRUG GM','MEAT','PRODUCE','KIOSK-GAS','MEAT-PCKGD','DELI','PASTRY','MISC SALES TRAN','NUTRITION']

def fn(view, snapshot_day):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions.merge(
        view.products[['product_id','department','brand']], on='product_id', how='left')
    out = pd.DataFrame(index=hh)
    # ---- 84-day department mix ----
    w84 = tx[tx.day > snapshot_day - 84]
    g = w84.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
    g = g.reindex(hh).fillna(0.0)
    tot = g.sum(axis=1).replace(0, np.nan)
    for d in DEPTS:
        out[f'share84_{d}'] = (g[d]/tot).fillna(0.0) if d in g.columns else 0.0
    out['div84'] = (g > 0).sum(axis=1).astype(float)
    out['max_share84'] = g.div(tot).max(axis=1).fillna(0.0)
    # ---- brand orientation ----
    b = w84.groupby(['household_key','brand']).sales_value.sum().unstack(fill_value=0.0)
    b = b.reindex(hh).fillna(0.0)
    btot = b.sum(axis=1).replace(0, np.nan)
    out['private_share84'] = (b['Private']/btot).fillna(0.0) if 'Private' in b.columns else 0.0
    # ---- 28-day per-department spend levels ----
    w28 = tx[tx.day > snapshot_day - 28]
    g28 = w28.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
    g28 = g28.reindex(hh).fillna(0.0)
    for d in DEPTS:
        out[f'spend28_{d}'] = g28[d] if d in g28.columns else 0.0
    return out

df1 = load_saved('e001_txhist.parquet')
bf = build_features(fn)
print("bf:", bf.shape, "nan cols:", int(bf.isna().any(axis=0).sum()))
# dtype alignment
for c in ['household_key','snapshot_day']:
    if df1[c].dtype != bf[c].dtype:
        print("casting", c, df1[c].dtype, "->", bf[c].dtype)
        df1[c] = df1[c].astype(bf[c].dtype)
m = df1.merge(bf.reset_index(), on=['household_key','snapshot_day'], how='inner')
print("merged:", m.shape, "(expect 36426)")
assert m.shape[0] == 36426 and m.notna().all().all()
path = save_table(m, 'e003_catmix')
print("saved:", path)


# ---- cell ----
import numpy as np, pandas as pd
df1 = load_saved('e001_txhist.parquet')
print("E001 nan counts:", df1.isna().sum()[lambda s: s>0].to_dict())
# reconstruct bf quickly to find collision
DEPTS = ['GROCERY','DRUG GM','MEAT','PRODUCE','KIOSK-GAS','MEAT-PCKGD','DELI','PASTRY','MISC SALES TRAN','NUTRITION']
def fn(view, snapshot_day):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions.merge(view.products[['product_id','department','brand']], on='product_id', how='left')
    out = pd.DataFrame(index=hh)
    w84 = tx[tx.day > snapshot_day - 84]
    g = w84.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
    tot = g.sum(axis=1).replace(0, np.nan)
    for d in DEPTS:
        out[f'share84_{d}'] = (g[d]/tot).fillna(0.0) if d in g.columns else 0.0
    out['div84'] = (g > 0).sum(axis=1).astype(float)
    out['max_share84'] = g.div(tot).max(axis=1).fillna(0.0)
    b = w84.groupby(['household_key','brand']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
    btot = b.sum(axis=1).replace(0, np.nan)
    out['private_share84'] = (b['Private']/btot).fillna(0.0) if 'Private' in b.columns else 0.0
    w28 = tx[tx.day > snapshot_day - 28]
    g28 = w28.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
    for d in DEPTS:
        out[f'spend28_{d}'] = g28[d] if d in g28.columns else 0.0
    return out
bf = build_features(fn).reset_index()
print("bf cols:", list(bf.columns))
print("overlap:", set(df1.columns) & set(bf.columns))


# ---- cell ----
import numpy as np, pandas as pd
DEPTS = ['GROCERY','DRUG GM','MEAT','PRODUCE','KIOSK-GAS','MEAT-PCKGD','DELI','PASTRY','MISC SALES TRAN','NUTRITION']
def fn(view, snapshot_day):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions.merge(view.products[['product_id','department','brand']], on='product_id', how='left')
    out = pd.DataFrame(index=hh)
    w84 = tx[tx.day > snapshot_day - 84]
    g = w84.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
    tot = g.sum(axis=1).replace(0, np.nan)
    for d in DEPTS:
        out[f'share84_{d}'] = (g[d]/tot).fillna(0.0) if d in g.columns else 0.0
    out['div84'] = (g > 0).sum(axis=1).astype(float)
    out['max_share84'] = g.div(tot).max(axis=1).fillna(0.0)
    b = w84.groupby(['household_key','brand']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
    btot = b.sum(axis=1).replace(0, np.nan)
    out['private_share84'] = (b['Private']/btot).fillna(0.0) if 'Private' in b.columns else 0.0
    w28 = tx[tx.day > snapshot_day - 28]
    g28 = w28.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
    for d in DEPTS:
        out[f'spend28_{d}'] = g28[d] if d in g28.columns else 0.0
    return out
df1 = load_saved('e001_txhist.parquet')
df1['evening_share84'] = df1['evening_share84'].fillna(0.0)
bf = build_features(fn)
m = df1.merge(bf, on=['household_key','snapshot_day'], how='inner')
print("merged:", m.shape, "nans:", int(m.isna().sum().sum()))
assert m.shape[0] == 36426 and not m.isna().any().any()
path = save_table(m, 'e003_catmix')
print("saved:", path)
