import pandas as pd, numpy as np

names = ["churn_vol_v1","combined_v1","comp_v1","demo_v1","ewma_block_v1","mkt_v1","rfm_cadence_v1","rfm_traj_v1","rfm_v1","season_demo_v1","union_all_v1"]
tabs = {}
for n in names:
    df = load_saved(n + ".parquet")
    tabs[n] = df
    print(n, df.shape)
    print("  ", list(df.columns))

tt = train_targets()
print("\ntargets:", tt.shape)
print(tt.future_spend_4w.describe())
print("\nmean target by snapshot_day:")
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"]))


# ---- cell ----
import pandas as pd, numpy as np

tt = train_targets()
cv = load_saved("churn_vol_v1.parquet")   # E010 table (best)
m = tt.merge(cv, on=["household_key","snapshot_day"], how="left")
y = m.future_spend_4w
feats = [c for c in cv.columns if c not in ("household_key","snapshot_day")]
corr = {}
for c in feats:
    v = m[c]
    if v.dtype == bool: v = v.astype(float)
    if not np.issubdtype(np.asarray(v).dtype, np.number):
        continue
    corr[c] = np.corrcoef(v.fillna(v.median()), y)[0,1]
cs = pd.Series(corr).sort_values(key=lambda s: s.abs(), ascending=False)
print(cs.head(45))
print("\n--- weakest |corr| ---")
print(cs.tail(15))


# ---- cell ----
import pandas as pd, numpy as np

def fn(view, day):
    tx = view.table("transactions")[["household_key","day","sales_value"]]
    hh = pd.Index(view.households)
    tx = tx.sort_values(["household_key","day"])
    keys = tx.household_key.values
    days = tx.day.values
    vals = tx.sales_value.values.astype(float)
    uniq, starts = np.unique(keys, return_index=True)
    starts = np.append(starts, len(keys))
    res = {}
    for i in range(len(uniq)):
        k = uniq[i]
        d = days[starts[i]:starts[i+1]]
        v = vals[starts[i]:starts[i+1]]
        cum = np.zeros(day+2)
        np.add.at(cum, d, v)
        cum = np.cumsum(cum)
        js = np.arange(1, (day-28)//28 + 1)   # window ends e = day-28j, need e>=28
        if len(js) == 0:
            continue
        ends = day - 28*js
        t = cum[ends] - cum[ends-28]          # trailing 28d spend ending at e
        f = cum[ends+28] - cum[ends]          # following 28d spend (fully observed)
        r = np.clip(f / np.maximum(t, 1.0), 0, 10)
        w = 0.5 ** (js.astype(float)/2.0)     # half-life 2 windows (56d), most recent j=1 heaviest
        w4 = 0.5 ** (js.astype(float)/4.0)
        def ew(x, wt):
            s = (x*wt).sum()
            return s / wt.sum() if wt.sum() > 0 else np.nan
        t_act = t > 0
        out = {
            "sc_n_pairs": len(js),
            "sc_f_mean": f.mean(), "sc_f_med": np.median(f), "sc_f_std": f.std(),
            "sc_f_max": f.max(), "sc_f_min": f.min(),
            "sc_f_mean3": f[js<=3].mean(), "sc_f_mean6": f[js<=6].mean(), "sc_f_mean12": f[js<=12].mean(),
            "sc_f_ew_hl2": ew(f, w), "sc_f_ew_hl4": ew(f, w4),
            "sc_f_ew_hl2_act": ew(f[t_act], w[t_act]) if t_act.any() else np.nan,
            "sc_ratio_mean": r.mean(), "sc_ratio_med": np.median(r), "sc_ratio_r3": r[js<=3].mean(),
            "sc_carry": np.clip(f.sum()/max(t.sum(),1.0), 0, 10),
            "sc_carry_act": np.clip(f[t_act].sum()/max(t[t_act].sum(),1.0), 0, 10) if t_act.any() else np.nan,
            "sc_f_zero_frac": (f==0).mean(),
            "sc_f_active_mean": f[t_act].mean() if t_act.any() else np.nan,
            "sc_t_mean": t.mean(),
            "sc_corr_tf": np.corrcoef(t, f)[0,1] if t.std()>0 and f.std()>0 else np.nan,
            "sc_f_13": f[js==13].mean() if (js==13).any() else np.nan,   # year-ago following window
            "sc_f_26": f[js==26].mean() if (js==26).any() else np.nan,
        }
        res[k] = out
    df = pd.DataFrame.from_dict(res, orient="index")
    df.index.name = "household_key"
    return df.reindex(hh)

blk = build_features(fn)
print(blk.shape)
print(blk.dropna(axis=1, how="all").shape)
print(blk.head(3).T)
save_table(blk, "selfcal_v1.parquet")


# ---- cell ----
import pandas as pd
base = load_saved("churn_vol_v1.parquet")
blk = load_saved("selfcal_v1.parquet")
out = base.merge(blk, on=["household_key","snapshot_day"], how="left")
print(out.shape)
save_table(out, "e011_table.parquet")
