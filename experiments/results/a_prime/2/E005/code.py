import agent_api
import pandas as pd
import numpy as np

def make_feats(view, sd):
    tx = view.transactions
    hh = pd.Index(view.households, name='household_key')
    out = pd.DataFrame(index=hh)

    # lifetime / tenure (tx is capped at sd, so this is point-in-time safe)
    g = tx.groupby('household_key')
    first_day = g.day.min().reindex(hh)
    tenure = (sd - first_day).clip(lower=1)
    out['tenure_days'] = tenure
    out['lifetime_spend'] = g.sales_value.sum().reindex(hh).fillna(0.0)
    out['lifetime_trips'] = g.basket_id.nunique().reindex(hh).fillna(0.0)
    out['lifetime_weekly'] = out['lifetime_spend'] / (tenure / 7.0)

    # 84-day window
    w = tx[tx.day > sd - 84]
    gw = w.groupby('household_key')
    spend84 = gw.sales_value.sum().reindex(hh).fillna(0.0)
    trips84 = gw.basket_id.nunique().reindex(hh).fillna(0.0)
    lines84 = gw.size().reindex(hh).fillna(0.0)
    qty84 = gw.quantity.sum().reindex(hh).fillna(0.0)
    out['trips_per_week_84'] = trips84 / 12.0
    out['accel_ratio'] = (spend84 / 12.0) / out['lifetime_weekly'].replace(0.0, np.nan)
    out['units_per_trip_84'] = qty84 / trips84.replace(0.0, np.nan)
    out['avg_unit_price_84'] = spend84 / qty84.replace(0.0, np.nan)

    if len(w):
        b = w.groupby(['household_key', 'basket_id']).agg(
            spend=('sales_value', 'sum'),
            lines=('product_id', 'size'),
            day=('day', 'first'),
            tt=('trans_time', 'mean'),
        ).reset_index()
        gb = b.groupby('household_key')
        out['basket_mean_84'] = gb.spend.mean().reindex(hh)
        out['basket_max_84'] = gb.spend.max().reindex(hh)
        out['basket_std_84'] = gb.spend.std().reindex(hh)
        out['lines_per_trip_84'] = gb.lines.mean().reindex(hh)
        out['spend_per_line_84'] = spend84 / lines84.replace(0.0, np.nan)
        out['distinct_products_84'] = gw.product_id.nunique().reindex(hh).fillna(0.0)
        out['distinct_stores_84'] = gw.store_id.nunique().reindex(hh).fillna(0.0)
        st = w.groupby(['household_key', 'store_id']).sales_value.sum().reset_index()
        stmax = st.groupby('household_key').sales_value.max().reindex(hh)
        out['top_store_share_84'] = stmax / spend84.replace(0.0, np.nan)
        dow = b.day % 7
        out['weekend_trip_share_84'] = (dow >= 5).groupby(b['household_key']).mean().reindex(hh)
        hour = b['tt'] // 100
        out['morning_trip_share_84'] = (hour < 12).groupby(b['household_key']).mean().reindex(hh)
    else:
        for c in ['basket_mean_84', 'basket_max_84', 'basket_std_84', 'lines_per_trip_84',
                  'spend_per_line_84', 'distinct_products_84', 'distinct_stores_84',
                  'top_store_share_84', 'weekend_trip_share_84', 'morning_trip_share_84']:
            out[c] = np.nan
    return out

tbl = agent_api.build_features(make_feats)
if 'household_key' not in tbl.columns:
    tbl = tbl.reset_index()
print('new feats:', tbl.shape)
print(tbl.columns.tolist())
print(tbl.drop(columns=['household_key','snapshot_day']).isna().mean().round(3).to_string())

base = agent_api.load_saved('temporal_structure.parquet')
print('base:', base.shape)
merged = base.merge(tbl, on=['household_key', 'snapshot_day'], how='left')
print('merged:', merged.shape)
path = agent_api.save_table(merged, 'basket_tenure.parquet')
print(path)
