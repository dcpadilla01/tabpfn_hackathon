import agent_api as A, pandas as pd, numpy as np

def fn(view, d):
    tx = view.table("transactions")
    tx = tx[tx.day <= d]
    hh = pd.Index(list(view.households), name="household_key")
    g = tx.groupby("household_key")
    out = pd.DataFrame(index=hh)
    def wsum(lo, hi):
        return tx[(tx.day > lo) & (tx.day <= hi)].groupby("household_key").sales_value.sum()
    for w in [7,14,28,56,84,180,365]:
        out[f"spend_{w}"] = wsum(d-w, d)
    out["spend_28_prior"] = wsum(d-56, d-28)
    out["spend_84_prior"] = wsum(d-168, d-84)
    for w in [28,84]:
        out[f"baskets_{w}"] = tx[tx.day > d-w].groupby("household_key").basket_id.nunique()
    out["days_since_last"] = d - g.day.max()
    out["days_since_first"] = d - g.day.min()
    out["avg_basket_84"] = out.spend_84/out.baskets_84.clip(lower=1)
    out["trips_per_wk_84"] = out.baskets_84/12.0
    out["spend_28_ratio"] = out.spend_28/out.spend_28_prior.clip(lower=1.0)
    t84 = tx[tx.day > d-84]
    out["n_products_84"] = t84.groupby("household_key").product_id.nunique()
    out["n_stores_84"] = t84.groupby("household_key").store_id.nunique()
    out["spend_trend"] = out.spend_28 - out.spend_28_prior
    out["active_28"] = out.spend_28 > 0
    age = (d - tx.day).astype(float)
    l2 = np.log(2)
    for hl in [7,14,28,56,84,180]:
        out[f"ew_{hl}"] = (tx.sales_value*np.exp(-age*l2/hl)).groupby(tx.household_key).sum()
    for k in [336,364,392]:
        out[f"spend_lag{k}"] = wsum(d-k-28, d-k)
    out["longrun_wk"] = out.spend_365/52.0
    out["ratio28_lr"] = out.spend_28/(out.longrun_wk*4).clip(lower=0.01)
    out["ratio84_lr"] = out.spend_84/(out.longrun_wk*12).clip(lower=0.01)
    b84 = t84.groupby(["household_key","basket_id"]).sales_value.sum()
    bb = b84.groupby(level=0)
    out["basket_max_84"] = bb.max(); out["basket_std_84"] = bb.std(); out["basket_med_84"] = bb.median()
    out["active_days_28"] = tx[tx.day > d-28].groupby("household_key").day.nunique()
    # NEW: trip-spacing volatility over 84d
    du = t84[["household_key","day"]].drop_duplicates().sort_values(["household_key","day"])
    gaps = du.groupby("household_key").day.diff()
    gg = gaps.groupby(du.household_key)
    out["gap_cv"] = gg.std()/gg.mean()
    return out

path = A.save_table(A.build_features(fn), "e005_decay_gapcv.parquet")
print(path)
