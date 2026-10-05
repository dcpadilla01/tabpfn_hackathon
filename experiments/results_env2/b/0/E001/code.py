import agent_api as A
print(A.describe_tables())
print(A.snapshot_days())
print(A.KEYS, A.TARGET)
v = A.snapshot(459)
tx = v.table("transactions")
print(tx.shape)
print(tx.head())
print("households needing row at 459:", len(v.households))
print(v.households[:5])


# ---- cell ----
import agent_api as A
v = A.snapshot(459)
print(type(v.households), v.day, v.week)
print([a for a in dir(v) if not a.startswith('_')])
# try history
h = A.history(2375, 459)
print(h.shape)
print(h.head())


# ---- cell ----
import agent_api as A
tt = A.train_targets()
print(tt.shape, tt.head())
print(tt.future_spend_4w.describe())
print("zero frac:", (tt.future_spend_4w==0).mean())
print(tt.groupby('snapshot_day').future_spend_4w.agg(['mean','median','count']))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

def make_feats(view, sd):
    hh = view.households
    tx = view.table("transactions")
    tx = tx[tx.day <= sd]
    g = tx.groupby('household_key')
    out = pd.DataFrame(index=hh)
    # spend in windows
    for w in [28, 56, 84, 112, 364]:
        s = tx[tx.day > sd - w].groupby('household_key').sales_value.sum()
        out[f'spend_{w}'] = s.reindex(hh).fillna(0.0)
    # trips
    trips = tx[tx.day > sd-28].drop_duplicates('basket_id').groupby('household_key').size()
    out['trips_28'] = trips.reindex(hh).fillna(0)
    trips56 = tx[tx.day > sd-56].drop_duplicates('basket_id').groupby('household_key').size()
    out['trips_56'] = trips56.reindex(hh).fillna(0)
    # recency
    last = g.day.max().reindex(hh)
    out['days_since_last'] = (sd - last).clip(lower=0).fillna(sd)
    out['active_days_28'] = tx[tx.day > sd-28].groupby('household_key').day.nunique().reindex(hh).fillna(0)
    # trend: last 28 vs prior 28
    p = tx[(tx.day > sd-56) & (tx.day <= sd-28)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
    c = out['spend_28']
    out['trend'] = (c - p) / (c + p + 1.0)
    # avg basket value last 56d
    bv = tx[tx.day > sd-56].groupby('basket_id').sales_value.sum()
    hhb = tx[tx.day > sd-56].drop_duplicates('basket_id').set_index('basket_id').household_key
    out['avg_basket_56'] = bv.groupby(hhb).mean().reindex(hh).fillna(0.0)
    # lifetime
    out['life_spend'] = g.sales_value.sum().reindex(hh).fillna(0.0)
    out['life_days'] = g.day.nunique().reindex(hh).fillna(0)
    # quantity
    out['qty_28'] = tx[tx.day > sd-28].groupby('household_key').quantity.sum().reindex(hh).fillna(0.0)
    # discounts
    for col in ['coupon_disc','retail_disc','coupon_match_disc']:
        out[col+'_28'] = tx[tx.day > sd-28].groupby('household_key')[col].sum().reindex(hh).fillna(0.0)
    # stores
    out['n_stores_28'] = tx[tx.day > sd-28].groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    # calendar
    out['snapshot_day'] = sd
    out['week'] = view.week
    return out

df = A.build_features(make_feats)
print(df.shape)
print(df.head())
path = A.save_table(df, 'e001_history')
print(path)
