import agent_api as A, pandas as pd, numpy as np

def fn(view, snapshot_day):
    t = view.table("transactions")
    hh = view.households
    d0 = snapshot_day
    g = t.groupby("household_key")
    def win(lo, hi):
        m = (t.day > d0 - lo) & (t.day <= d0 - hi)  # spend in (d0-lo, d0-hi]
        return t.sales_value.where(m, 0).groupby(t.household_key).sum()
    f = pd.DataFrame(index=hh)
    for w in (28, 56, 84, 180, 365):
        f[f"spend_{w}"] = win(w, 0).reindex(hh).fillna(0)
    f["spend_28_prior"] = win(56, 28).reindex(hh).fillna(0)
    f["spend_84_prior"] = win(168, 84).reindex(hh).fillna(0)
    f["baskets_28"] = t[t.day > d0-28].groupby("household_key").basket_id.nunique().reindex(hh).fillna(0)
    f["baskets_84"] = t[t.day > d0-84].groupby("household_key").basket_id.nunique().reindex(hh).fillna(0)
    f["days_since_last"] = d0 - g.day.max().reindex(hh)
    f["days_since_first"] = d0 - g.day.min().reindex(hh)
    f["avg_basket_84"] = f.spend_84 / f.baskets_84.replace(0, np.nan)
    f["trips_per_wk_84"] = f.baskets_84 / 12.0
    f["spend_28_ratio"] = f.spend_28 / (f.spend_28 + f.spend_28_prior + 1e-9)
    f["n_products_84"] = t[t.day > d0-84].groupby("household_key").product_id.nunique().reindex(hh).fillna(0)
    f["n_stores_84"] = t[t.day > d0-84].groupby("household_key").store_id.nunique().reindex(hh).fillna(0)
    f["spend_trend"] = f.spend_28 - f.spend_28_prior
    f["active_28"] = (f.spend_28 > 0).astype(int)
    return f

tab = A.build_features(fn)
print(tab.shape, tab.columns.tolist())
print(tab.head())
path = A.save_table(tab, "e001_recent_behavior")
print(path)
