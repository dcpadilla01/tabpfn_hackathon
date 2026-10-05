import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved("rfm_cadence_v1.parquet")
demo = agent_api.load_saved("demo_v1.parquet")
print("base", base.shape, list(base.columns)[:10])
print("demo", demo.shape, list(demo.columns)[:10])
t = agent_api.train_targets()
print("target n:", len(t), "zero-frac: %.3f" % (t.future_spend_4w == 0).mean())
print(t.future_spend_4w.describe().round(2))

def feats(view, snapshot_day):
    day = int(snapshot_day)
    tx = view.table("transactions")
    tx = tx[tx["day"] <= day]
    h = view.households
    hh = pd.Index(h["household_key"]) if isinstance(h, pd.DataFrame) else pd.Index(h)
    out = pd.DataFrame(index=hh)
    age = (day - tx["day"]).astype(float)

    # EWMA spend / trips at several half-lives
    bt = tx.drop_duplicates("basket_id")[["household_key", "day"]]
    for H in (14, 28, 56, 112):
        w = np.power(0.5, age.values / H)
        out["ew_spend_h%d" % H] = tx.assign(_w=tx["sales_value"].values * w).groupby("household_key")["_w"].sum()
        bw = np.power(0.5, (day - bt["day"]).values / H)
        out["ew_trips_h%d" % H] = bt.assign(_w=bw).groupby("household_key")["_w"].sum()

    # fine-grained recent windows
    for W in (7, 14):
        m = tx["day"] > day - W
        out["spend_%dd" % W] = tx[m].groupby("household_key")["sales_value"].sum()
        out["trips_%dd" % W] = tx[m].drop_duplicates("basket_id").groupby("household_key").size()

    # 13 aligned 28-day blocks over past 364d (block 0 = most recent, 12 = year ago)
    blk = ((day - tx["day"]) // 28).clip(upper=12)
    tb = tx.assign(blk=blk).groupby(["household_key", "blk"])["sales_value"].sum().unstack(fill_value=0.0)
    tb = tb.reindex(columns=range(13), fill_value=0.0).reindex(hh, fill_value=0.0)
    vals = tb.values.astype(float)
    out["spend_28d"] = vals[:, 0]
    out["blk_mean"] = vals.mean(axis=1)
    out["blk_median"] = np.median(vals, axis=1)
    out["blk_std"] = vals.std(axis=1)
    out["blk_min"] = vals.min(axis=1)
    out["blk_max"] = vals.max(axis=1)
    out["blk_cv"] = vals.std(axis=1) / (vals.mean(axis=1) + 1.0)
    out["blk_active_frac"] = (vals > 0).mean(axis=1)
    out["blk_active_frac6"] = (vals[:, :6] > 0).mean(axis=1)
    out["blk_last_over_mean"] = vals[:, 0] / (vals.mean(axis=1) + 1.0)
    out["blk_year_ago"] = vals[:, 12]
    xc = np.arange(13, dtype=float); xc -= xc.mean()
    out["blk_slope"] = (vals * xc).sum(axis=1) / (xc ** 2).sum()
    z = (vals == 0)
    def lz(row):
        best = cur = 0
        for v in row:
            cur = 0 if v else cur + 1
            if cur > best: best = cur
        return best
    out["blk_zero_run"] = [lz(r) for r in z]

    # cadence / breadth
    out["days_since_last"] = day - tx.groupby("household_key")["day"].max()
    m84 = tx["day"] > day - 84
    out["n_stores_84d"] = tx[m84].groupby("household_key")["store_id"].nunique()
    m28 = tx["day"] > day - 28
    out["active_days_28d"] = tx[m28].groupby("household_key")["day"].nunique()
    ad = out["active_days_28d"].fillna(0).values
    out["spend_per_active_day_28d"] = vals[:, 0] / (ad + 1.0)

    # global seasonality (mod-52 so validation weeks stay in train range)
    wk = ((day + 8) // 7) % 52
    out["week_of_year"] = wk
    out["week_sin"] = np.sin(2 * np.pi * wk / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * wk / 52.0)
    return out

tbl = agent_api.build_features(feats)
if "household_key" not in tbl.columns:
    tbl = tbl.reset_index()
print("built:", tbl.shape, "n_new_feats:", tbl.shape[1] - 2)

m = base.merge(demo, on=["household_key", "snapshot_day"], how="left")
m = m.merge(tbl, on=["household_key", "snapshot_day"], how="left")
print("merged:", m.shape)
print("dup key rows:", m.duplicated(["household_key", "snapshot_day"]).sum())
print("nulls ew_spend_h28:", int(m["ew_spend_h28"].isna().sum()), "| nulls blk_mean:", int(m["blk_mean"].isna().sum()))
path = agent_api.save_table(m, "ewma_block_v1")
print("saved:", path)
