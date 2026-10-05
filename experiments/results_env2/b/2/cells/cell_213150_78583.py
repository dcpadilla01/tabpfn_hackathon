import numpy as np, pandas as pd, agent_api

saved = agent_api.load_saved("rfm_traj_v1.parquet")

def build(view, S):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    out = pd.DataFrame(index=hh)
    tx364 = tx[tx["day"] > S - 364]
    first_day = tx.groupby("household_key")["day"].min()
    out["tenure_days"] = (S - first_day).reindex(hh).astype(float)
    if len(tx364) > 0:
        b = (tx364.groupby(["household_key", "basket_id"])
                  .agg(day=("day", "min"), val=("sales_value", "sum"))
                  .reset_index()
                  .sort_values(["household_key", "day"]))
        b["gap"] = b.groupby("household_key")["day"].diff()
        gg = b.groupby("household_key")
        out["trips_364"] = gg.size().reindex(hh).astype(float)
        out["basket_mean_364"] = gg["val"].mean().reindex(hh)
        out["basket_std_364"] = gg["val"].std().reindex(hh)
        out["gap_mean"] = gg["gap"].mean().reindex(hh)
        out["gap_std"] = gg["gap"].std().reindex(hh)
        out["gap_max"] = gg["gap"].max().reindex(hh)
        out["spend_364"] = tx364.groupby("household_key")["sales_value"].sum().reindex(hh)
        ws = (tx364.assign(wk=tx364["day"] // 7)
                    .groupby(["household_key", "wk"])["sales_value"].sum().reset_index())
        wg = ws.groupby("household_key")["sales_value"]
        out["wk_spend_mean"] = wg.mean().reindex(hh)
        out["wk_spend_std"] = wg.std().reindex(hh)
        out["wk_spend_med"] = wg.median().reindex(hh)
        out["active_weeks_364"] = ws.groupby("household_key").size().reindex(hh).astype(float)
        age = (S - tx364["day"]).astype(float)
        hk = tx364["household_key"]
        for hl in (14, 28, 56, 112):
            wgt = np.power(0.5, age / hl)
            out["ew_spend_hl%d" % hl] = (tx364["sales_value"] * wgt).groupby(hk).sum().reindex(hh)
    for c in ["trips_364","basket_mean_364","basket_std_364","gap_mean","gap_std","gap_max",
              "spend_364","wk_spend_mean","wk_spend_std","wk_spend_med","active_weeks_364",
              "ew_spend_hl14","ew_spend_hl28","ew_spend_hl56","ew_spend_hl112"]:
        if c not in out:
            out[c] = np.nan
    tw = (out["tenure_days"] / 7.0).clip(lower=1.0)
    out["active_week_ratio"] = out["active_weeks_364"] / np.minimum(52.0, tw)
    out["spend_per_trip_364"] = out["spend_364"] / out["trips_364"]
    out["gap_cv"] = out["gap_std"] / out["gap_mean"]
    return out

feat = agent_api.build_features(build)
wcols = ["spend_w2","spend_w3","spend_w4","spend_w5","spend_w6"]
usual = saved[wcols].mean(axis=1)
saved = saved.copy()
saved["usual_4w"] = usual
saved["ratio_recent_usual"] = saved["spend_w1"] / usual.replace(0, np.nan)

m = saved.merge(feat, on=["household_key", "snapshot_day"], how="inner")
assert len(m) == len(saved), (len(m), len(saved))
print(m[["usual_4w","ratio_recent_usual","gap_mean","gap_cv","active_week_ratio",
         "ew_spend_hl28","tenure_days","wk_spend_std"]].describe().T)
path = agent_api.save_table(m, "rfm_cadence_v1.parquet")
print("saved to", path)
