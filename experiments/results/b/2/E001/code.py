
import agent_api as api
import numpy as np, pandas as pd

def fn(view, snapshot_day):
    t = view.transactions
    hh = view.households
    d = snapshot_day
    g = t.groupby('household_key')
    agg = pd.DataFrame(index=hh)
    # spend windows
    for w in (28, 56, 84, 112, 364):
        agg[f'spend_{w}'] = t[t.day > d - w].groupby('household_key').sales_value.sum().reindex(hh, fill_value=0.0)
    # aligned prior windows (same length as target window)
    agg['spend_lag1'] = t[(t.day > d-56) & (t.day <= d-28)].groupby('household_key').sales_value.sum().reindex(hh, fill_value=0.0)
    agg['spend_lag2'] = t[(t.day > d-84) & (t.day <= d-56)].groupby('household_key').sales_value.sum().reindex(hh, fill_value=0.0)
    # trips
    trips28 = t[t.day > d-28].groupby('household_key').basket_id.nunique().reindex(hh, fill_value=0)
    trips56 = t[t.day > d-56].groupby('household_key').basket_id.nunique().reindex(hh, fill_value=0)
    agg['trips_28'] = trips28
    agg['trips_56'] = trips56
    agg['basket_avg_28'] = agg['spend_28'] / trips28.replace(0, np.nan)
    agg['active_days_28'] = t[t.day > d-28].groupby('household_key').day.nunique().reindex(hh, fill_value=0)
    # lifetime
    first = g.day.min().reindex(hh)
    last = g.day.max().reindex(hh)
    agg['tenure'] = (d - first).clip(lower=0)
    agg['recency'] = (d - last).clip(lower=0)
    agg['spend_life'] = g.sales_value.sum().reindex(hh, fill_value=0.0)
    agg['trips_life'] = g.basket_id.nunique().reindex(hh, fill_value=0)
    agg['spend_rate_life'] = agg['spend_life'] / agg['tenure'].replace(0, np.nan)
    # trend & momentum
    agg['trend'] = agg['spend_28'] - agg['spend_lag1']
    agg['ratio_lag'] = agg['spend_28'] / agg['spend_lag1'].replace(0, np.nan)
    agg['wk_avg_84'] = agg['spend_84'] / 12.0
    return agg

out = api.build_features(fn)
print(out.shape)
print(out.head())
path = api.save_table(out, 'e001_recent_spend')
print(path)
