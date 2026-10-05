
import pandas as pd, numpy as np

e10 = agent_api.load_saved("e010_lifecycle.parquet")
print("saved", e10.shape)

def build(view, snapshot_day):
    hh = pd.Index(np.asarray(view.households), name="household_key")
    tx = view.table("transactions")
    tx = tx[tx["household_key"].isin(set(hh))].copy()
    tx["_disc"] = tx["coupon_disc"] + tx["coupon_match_disc"] + tx["retail_disc"]
    out = pd.DataFrame(index=hh)
    for W, suf in [(84, "84"), (364, "364")]:
        t = tx[tx["day"] > snapshot_day - W]
        g = t.groupby("household_key")
        sp = g["sales_value"].sum().reindex(hh).fillna(0.0)
        ds = g["_disc"].sum().reindex(hh).fillna(0.0)
        out["rd_" + suf] = g["retail_disc"].sum().reindex(hh).fillna(0.0)
        out["cd_" + suf] = g["coupon_disc"].sum().reindex(hh).fillna(0.0)
        out["disc_share_" + suf] = ((-ds) / (sp - ds).replace(0, np.nan)).fillna(0.0)
        cb = t[t["coupon_disc"] < 0].groupby("household_key")["basket_id"].nunique().reindex(hh).fillna(0.0)
        nb = g["basket_id"].nunique().reindex(hh).fillna(0.0)
        out["coup_trips_" + suf] = cb
        if suf == "364":
            out["coup_trip_share_364"] = (cb / nb.replace(0, np.nan)).fillna(0.0)
            out["net_spend_364"] = sp + ds
            out["units_per_trip_364"] = np.log1p(g["quantity"].sum().reindex(hh).fillna(0.0) / nb.replace(0, np.nan))
        else:
            out["coup_trips_84"] = cb
    cr = view.table("coupon_redemptions")
    cr = cr[cr["household_key"].isin(set(hh))]
    gcr = cr.groupby("household_key")
    out["days_since_redem"] = (snapshot_day - gcr["day"].max()).reindex(hh).fillna(999.0)
    out["redem_84"] = cr[cr["day"] > snapshot_day - 84].groupby("household_key")["day"].count().reindex(hh).fillna(0.0)
    return out

feats = agent_api.build_features(build)
print("feats", feats.shape, list(feats.columns))
print(feats.head(3).to_string())

mrg = e10.merge(feats.reset_index(), on=["household_key", "snapshot_day"], how="left")
print("merged", mrg.shape)
newcols = [c for c in feats.reset_index().columns if c not in ("household_key", "snapshot_day")]
print("n new", len(newcols), "total feat cols", mrg.shape[1] - 2)
print("NaN counts in new cols:\n", mrg[newcols].isna().sum().to_string())

tt = agent_api.train_targets()
tr = mrg.merge(tt, on=["household_key", "snapshot_day"])
print("corr with target:")
for c in newcols:
    print(f"  {c}: {np.corrcoef(tr[c].fillna(tr[c].median()), tr['future_spend_4w'])[0,1]:.3f}")

path = agent_api.save_table(mrg, "e011_discounts.parquet")
print("saved", path)
