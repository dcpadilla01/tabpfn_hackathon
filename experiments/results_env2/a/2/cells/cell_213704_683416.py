
import agent_api as api, pandas as pd, numpy as np

def make_feats(view, snapshot_day):
    tr = view.table("transactions")
    hh = pd.Index(view.households, name="household_key")
    out = pd.DataFrame(index=hh)
    d = tr[tr.day <= snapshot_day]
    g = d.groupby('household_key')

    for w in [28, 56, 84, 112, 224]:
        s = d[d.day > snapshot_day - w].groupby('household_key').sales_value.sum()
        out[f'spend_{w}'] = s.reindex(hh).fillna(0.0)
        b = d[d.day > snapshot_day - w].groupby('household_key').basket_id.nunique()
        out[f'baskets_{w}'] = b.reindex(hh).fillna(0.0)

    # trend ratios
    out['trend_28_56'] = out['spend_28'] / (out['spend_56']/2 + 1)
    out['trend_56_112'] = out['spend_56'] / (out['spend_112']/2 + 1)
    # per-basket value
    out['abv_28'] = out['spend_28'] / (out['baskets_28'] + 1)
    # recency & tenure
    out['days_since_last'] = snapshot_day - g.day.max().reindex(hh)
    out['days_since_first'] = snapshot_day - g.day.min().reindex(hh)
    out['active_28'] = (out['baskets_28'] > 0).astype(float)
    # discount usage last 84d
    d84 = d[d.day > snapshot_day - 84]
    out['coupon_disc_84'] = d84.groupby('household_key').coupon_disc.sum().reindex(hh).fillna(0)
    out['retail_disc_84'] = d84.groupby('household_key').retail_disc.sum().reindex(hh).fillna(0)
    # distinct stores/products last 84d
    out['n_stores_84'] = d84.groupby('household_key').store_id.nunique().reindex(hh).fillna(0)
    out['n_prods_84'] = d84.groupby('household_key').product_id.nunique().reindex(hh).fillna(0)
    # demographics
    demo = view.table("demographics")
    if demo is not None and len(demo):
        demo = demo.set_index('household_key')
        for c in demo.columns:
            out['dem_'+c] = demo[c].reindex(hh).astype('category').cat.codes.replace(-1, np.nan)
    out['has_demo'] = out.index.isin(demo.index).astype(float) if demo is not None and len(demo) else 0.0
    # calendar
    out['snapshot_day'] = float(snapshot_day)
    out['week_of_year'] = ((snapshot_day + 8) // 7) % 52
    return out

feats = api.build_features(make_feats)
print(feats.shape, feats.columns.tolist())
print(feats.head(3))
api.save_table(feats, "feats_v1")
