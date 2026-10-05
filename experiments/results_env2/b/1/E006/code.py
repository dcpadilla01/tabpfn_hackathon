
import numpy as np, pandas as pd, agent_api

e003 = agent_api.load_saved("e003_full.parquet")
print("e003 shape:", e003.shape)
print("n cols:", len(e003.columns))
print(sorted(e003.columns))

def feats(view, s):
    tx = view.table("transactions")
    hh = pd.Index(view.households)
    # trip-level table
    t = tx.groupby(["household_key","basket_id"], as_index=False).agg(
        day=("day","max"), val=("sales_value","sum"), tt=("trans_time","mean"))
    g = t.groupby("household_key")
    out = pd.DataFrame(index=hh)

    out["cd_days_since_last"] = s - g["day"].max()
    out["cd_trips_total"] = g.size()

    def _gap(x, f):
        d = np.sort(pd.unique(x))
        if len(d) < 2: return np.nan
        return f(np.diff(d))
    out["cd_med_gap"]  = g["day"].agg(lambda x: _gap(x, np.median))
    out["cd_mean_gap"] = g["day"].agg(lambda x: _gap(x, np.mean))
    out["cd_std_gap"]  = g["day"].agg(lambda x: _gap(x, np.std))
    out["cd_exp_trips"] = 28.0 / out["cd_med_gap"]
    out["cd_gap_cv"] = out["cd_std_gap"] / out["cd_mean_gap"]

    for w in (28, 56, 84, 168):
        out[f"cd_trips_{w}"] = g["day"].agg(lambda x, w=w: (x > s - w).sum())
    out["cd_active_share_84"] = out["cd_trips_84"] / 84.0
    out["cd_no_trip_28"] = (out["cd_trips_28"] == 0).astype(float)

    b = t[t.day > s - 168]
    gb = b.groupby("household_key")["val"]
    out["cd_bval_mean"] = gb.mean()
    out["cd_bval_med"]  = gb.median()
    out["cd_bval_std"]  = gb.std()
    out["cd_bval_max"]  = gb.max()
    out["cd_pred_spend"] = out["cd_exp_trips"] * out["cd_bval_mean"]

    # seasonal: same 4-week window one year earlier
    seas = tx[(tx.day >= s - 363) & (tx.day <= s - 336)]
    out["cd_seas_spend"] = seas.groupby("household_key")["sales_value"].sum()
    out["cd_seas_trips"] = seas.groupby("household_key")["basket_id"].nunique()
    sp28 = tx[tx.day > s - 28].groupby("household_key")["sales_value"].sum()
    out["cd_ratio_sp28_seas"] = sp28 / (out["cd_seas_spend"].fillna(0) + 1.0)

    # discount share over 364d
    d = tx[tx.day > s - 364]
    disc = (d["coupon_disc"].fillna(0) + d["coupon_match_disc"].fillna(0)
            + d["retail_disc"].fillna(0)).groupby(d["household_key"]).sum()
    sv = d.groupby("household_key")["sales_value"].sum()
    out["cd_disc_share"] = disc / (disc + sv).replace(0, np.nan)

    # evening shopping share (last 168d)
    b168 = t[t.day > s - 168].copy()
    b168["eve"] = (b168["tt"] >= 1700).astype(float)
    out["cd_evening_share"] = b168.groupby("household_key")["eve"].mean()

    # weekday-residue mix of trips (last 168d)
    b168["res"] = b168["day"] % 7
    ct = pd.crosstab(b168["household_key"], b168["res"], normalize="index")
    ct.columns = [f"cd_res_{c}" for c in ct.columns]
    out = out.join(ct)

    return out.reindex(hh)

feat_all = agent_api.build_features(feats)
print("feat_all:", feat_all.shape, "n new feats:", feat_all.shape[1] - 2)
full = e003.merge(feat_all, on=["household_key", "snapshot_day"], how="left")
print("full:", full.shape)
path = agent_api.save_table(full, "e006_cadence.parquet")
print("saved:", path)
