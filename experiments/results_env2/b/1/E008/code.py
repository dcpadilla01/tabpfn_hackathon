import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import agent_api

e003 = agent_api.load_saved("e003_full.parquet")
print("e003:", e003.shape)

tt = agent_api.train_targets()
print(tt.future_spend_4w.describe())
print("zero share train:", float((tt.future_spend_4w == 0).mean()))

def fn(view, snapshot_day):
    day = int(view.day)
    hh = pd.Index(view.households)
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(set(hh))].sort_values("day")
    darr = tx["day"].values
    f = pd.DataFrame(index=hh)

    gmin = tx.groupby("household_key")["day"].min().reindex(hh)
    gmax = tx.groupby("household_key")["day"].max().reindex(hh)
    tot = tx.groupby("household_key")["sales_value"].sum().reindex(hh)
    span = (day - gmin).astype(float)
    f["z_tenure_days"] = span
    f["z_recency_days"] = (day - gmax).astype(float)
    f["z_mean_week_spend_all"] = tot / np.clip(span / 7.0, 1.0, None)

    def win_sum(lo, hi):
        i0 = np.searchsorted(darr, lo, side="left")
        i1 = np.searchsorted(darr, hi, side="right")
        if i1 <= i0:
            return pd.Series(0.0, index=hh)
        s = tx.iloc[i0:i1].groupby("household_key")["sales_value"].sum()
        return s.reindex(hh).fillna(0.0)

    # weekly spend trajectory, last 12 weeks
    wk = np.zeros((len(hh), 12))
    for k in range(12):
        wk[:, k] = win_sum(day - 7*(k+1) + 1, day - 7*k).values
    for k in range(12):
        f[f"z_wk_sp_{k}"] = wk[:, k]
    wm = wk.mean(axis=1); ws = wk.std(axis=1)
    f["z_cv_week12"] = ws / np.clip(wm, 0.01, None)
    f["z_active_weeks12"] = (wk > 0).sum(axis=1) / 12.0

    # log-transformed window sums (heavy-tailed target)
    for w in (28, 56, 84, 168, 364):
        f[f"z_log_sp{w}"] = np.log1p(win_sum(day - w + 1, day).values)
    sp28 = win_sum(day - 27, day).values
    f["z_sp_last3d"] = win_sum(day - 2, day).values
    f["z_active_last28"] = sp28 > 0

    # aligned 4-week windows over full history: robust level + lapse rate
    ks = np.arange(27)
    los = day - 28*ks - 27
    W = np.zeros((len(hh), 27))
    for j in range(27):
        W[:, j] = win_sum(los[j], los[j] + 27).values
    V = span.values[:, None] >= (28*ks + 27)[None, :]
    Wm = pd.DataFrame(np.where(V, W, np.nan), index=hh)
    f["z_med4w_hist"] = Wm.median(axis=1)
    f["z_max4w_hist"] = Wm.max(axis=1)
    nz = V.sum(axis=1)
    f["z_zero_rate_hist"] = np.where(nz > 0, ((W == 0) & V).sum(axis=1) / np.clip(nz, 1, None), np.nan)
    f["z_n_windows_hist"] = nz
    return f

feat = agent_api.build_features(fn)
print("feat:", feat.shape)
merged = e003.merge(feat, on=list(agent_api.KEYS), how="inner")
print("merged:", merged.shape)
path = agent_api.save_table(merged, "e008_level_shape.parquet")
print("saved:", path)
