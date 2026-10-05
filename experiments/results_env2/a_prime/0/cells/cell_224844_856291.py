import agent_api
from agent_api import build_features, save_table, load_saved
import numpy as np, pandas as pd

e7 = load_saved("e007_te.parquet")
print("e7", e7.shape, "| dup keys:", e7.duplicated(["household_key","snapshot_day"]).sum())
print("e7 key dtypes:", e7["household_key"].dtype, e7["snapshot_day"].dtype)
print("e7 cols:", list(e7.columns[:8]), "...")

HALVES = [7, 14, 28, 56]

def fn(view, snapshot_day):
    hh = view.households
    if isinstance(hh, pd.DataFrame):
        hh = pd.Index(hh["household_key"] if "household_key" in hh.columns else hh.iloc[:, 0])
    hh = pd.Index(hh)
    if hh.name is None:
        hh.name = "household_key"
    t = view.table("transactions")
    t = t[t["day"] <= snapshot_day]
    out = pd.DataFrame(index=hh)
    if len(t):
        daily = (t.groupby(["household_key", "day"], as_index=False)
                  .agg(spend=("sales_value", "sum"), baskets=("basket_id", "nunique")))
        daily["log_spend"] = np.log1p(daily["spend"].clip(lower=0).to_numpy())
        days = daily["day"].to_numpy()
        first = daily.groupby("household_key")["day"].min().reindex(hh)
        L = np.clip(snapshot_day - first.to_numpy(dtype=float), 0, None)
        for h in HALVES:
            d = 0.5 ** (1.0 / h)
            w = np.power(d, (snapshot_day - days).astype(float))
            tmp = pd.DataFrame({"hh": daily["household_key"].to_numpy(), "w": w,
                                "ws": daily["spend"].to_numpy() * w,
                                "wl": daily["log_spend"].to_numpy() * w,
                                "wb": daily["baskets"].to_numpy() * w,
                                "wd": days * w})
            g = tmp.groupby("hh", as_index=False).agg(
                dec_days=("w", "sum"), dec_sum=("ws", "sum"), dec_log=("wl", "sum"),
                dec_bask=("wb", "sum"), wday=("wd", "sum")).set_index("hh").reindex(hh)
            span = (1.0 - np.power(d, L + 1.0)) / (1.0 - d)
            dd = g["dec_days"].replace(0.0, np.nan)
            out[f"ewm_spend_hl{h}"] = g["dec_sum"]
            out[f"ewm_logspend_hl{h}"] = g["dec_log"]
            out[f"ewm_spend_rate_hl{h}"] = g["dec_sum"] / span
            out[f"ewm_spend_avg_hl{h}"] = g["dec_sum"] / dd
            out[f"ewm_bask_rate_hl{h}"] = g["dec_bask"] / span
            out[f"ewm_actdays_hl{h}"] = g["dec_days"]
            out[f"ewm_recency_hl{h}"] = g["wday"] / dd
    out["ewm_mom7_56"] = out["ewm_spend_rate_hl7"] / (out["ewm_spend_rate_hl56"] + 1e-6)
    out["ewm_mom14_56"] = out["ewm_spend_rate_hl14"] / (out["ewm_spend_rate_hl56"] + 1e-6)
    out["ewm_mom28_56"] = out["ewm_spend_rate_hl28"] / (out["ewm_spend_rate_hl56"] + 1e-6)
    out = out.replace([np.inf, -np.inf], np.nan)
    if snapshot_day <= 151:
        print("snap", snapshot_day, out.shape, "nan%%=%.4f" % out.isna().mean().mean())
    return out

ewm = build_features(fn)
print("ewm", ewm.shape, "dups:", ewm.duplicated(["household_key","snapshot_day"]).sum())
m = e7.merge(ewm, on=["household_key", "snapshot_day"], how="left", validate="one_to_one")
print("merged", m.shape, "n_feat:", m.shape[1] - 2)
assert m.shape[0] == e7.shape[0] == 36426
p = save_table(m, "e010_ewma.parquet")
print("saved", p)
