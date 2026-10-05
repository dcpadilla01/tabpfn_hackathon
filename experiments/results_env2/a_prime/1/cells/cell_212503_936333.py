
import numpy as np, pandas as pd

def fn(view, snapshot_day):
    tx = view.table("transactions")
    hh = pd.Index(view.households, name="household_key")
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby("household_key")

    out = pd.DataFrame(index=hh)
    for w in [28, 56, 84, 112, 182, 365]:
        t = tx[tx.day > snapshot_day - w]
        s = t.groupby("household_key").sales_value.sum()
        out[f"spend_{w}"] = s.reindex(hh).fillna(0.0)
        out[f"trips_{w}"] = t.groupby("household_key").basket_id.nunique().reindex(hh).fillna(0.0)
        out[f"days_{w}"] = t.groupby("household_key").day.nunique().reindex(hh).fillna(0.0)

    out["recency"] = snapshot_day - g.day.max().reindex(hh)
    out["tenure"] = snapshot_day - g.day.min().reindex(hh)
    out["basket_mean_84"] = out["spend_84"] / out["trips_84"].replace(0, np.nan)
    out["items_mean_84"] = g.quantity.sum().reindex(hh) / out["tenure"].clip(lower=1) * 7
    out["trend_28_84"] = out["spend_28"] * 3 / out["spend_84"].replace(0, np.nan)
    out["spend_rate_84"] = out["spend_84"] / 84 * 28
    out["spend_rate_182"] = out["spend_182"] / 182 * 28
    out["disc_share_84"] = (-tx[tx.day > snapshot_day-84].groupby("household_key")["retail_disc"].sum()).reindex(hh).fillna(0.0) / out["spend_84"].replace(0, np.nan)
    out["stores_84"] = tx[tx.day > snapshot_day-84].groupby("household_key").store_id.nunique().reindex(hh).fillna(0.0)
    return out

tab = agent_api.build_features(fn)
print(tab.shape)
print(tab.head())
path = agent_api.save_table(tab, "e001_rfm")
print(path)
