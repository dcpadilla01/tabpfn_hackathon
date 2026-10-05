import numpy as np, pandas as pd, warnings, re

def fn(view, snapshot_day):
    tx = view.table("transactions")[["household_key","basket_id","day","sales_value"]].copy()
    tx["k"] = (snapshot_day - tx["day"]) // 28
    tx = tx[tx["k"] <= 17]
    wsum  = tx.groupby(["household_key","k"])["sales_value"].sum().unstack(fill_value=0.0).reindex(columns=range(18), fill_value=0.0)
    first_day = tx.groupby("household_key")["day"].min()
    idx = wsum.index
    W  = wsum.values.astype(float)
    fd = first_day.reindex(idx).values.astype(float)
    kmax = np.floor((snapshot_day - 27 - fd) / 28.0)
    ks = np.arange(18)
    valid = (ks[None,:] >= 1) & (ks[None,:] <= 12) & (ks[None,:] <= kmax[:,None])
    Wv = np.where(valid, W, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_mean = np.nanmean(Wv, axis=1); lr_med = np.nanmedian(Wv, axis=1)
        lr_std  = np.nanstd(Wv, axis=1);  lr_max = np.nanmax(Wv, axis=1)
        lr_min  = np.nanmin(Wv, axis=1)
        late_m  = np.nanmean(np.where(valid & (ks[None,:]>=7), W, np.nan), axis=1)
        early_m = np.nanmean(np.where(valid & (ks[None,:]<=6), W, np.nan), axis=1)
        denom   = W[:,0] + np.nansum(Wv, axis=1)
        conc3   = (W[:,0] + np.nansum(np.where(valid[:,1:3], W[:,1:3], np.nan), axis=1)) / np.where(denom>0, denom, np.nan)
    n_valid = valid.sum(axis=1).astype(float)
    lr_active = (np.where(valid, W > 0, False)).sum(axis=1).astype(float)
    win_cv = lr_std / np.where(lr_mean > 0, lr_mean, np.nan)
    elr = early_m / np.where(late_m > 0, late_m, np.nan)
    slope = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        m = valid[i]
        if m.sum() >= 4:
            x = ks[m].astype(float); y = W[i, m]
            xm = x - x.mean(); ym = y - y.mean()
            d = (xm**2).sum()
            if d > 0: slope[i] = (xm*ym).sum()/d
    t364 = tx[(tx["day"] > snapshot_day-364) & (tx["day"] <= snapshot_day)]
    dpu  = t364.drop_duplicates(["household_key","day"]).sort_values(["household_key","day"])
    g = dpu.groupby("household_key")["day"]
    internal = g.diff().groupby(dpu["household_key"]).max()
    f = g.first(); l = g.last()
    bnd = np.maximum(f - (snapshot_day-363), snapshot_day - l)
    max_gap = pd.concat([internal, bnd], axis=1).max(axis=1)
    trips_364  = t364.groupby("household_key")["basket_id"].nunique()
    adays_364  = g.count()
    spend_364  = t364.groupby("household_key")["sales_value"].sum()
    out = pd.DataFrame({
        "lr_mean28": lr_mean, "lr_med28": lr_med, "lr_std28": lr_std, "lr_max28": lr_max,
        "lr_min28": lr_min, "lr_active_wins": lr_active, "n_valid_wins": n_valid,
        "win_cv": win_cv, "slope_13w": slope, "conc3": conc3, "early_late_ratio": elr,
        "trips_364": trips_364.reindex(idx).values, "active_days_364": adays_364.reindex(idx).values,
        "spend_364": spend_364.reindex(idx).values, "max_gap_364": max_gap.reindex(idx).values,
    }, index=idx)
    dem = view.table("demographics").set_index("household_key")
    def num(s):
        try: return float(re.findall(r"\d+", str(s))[0])
        except Exception: return np.nan
    out["demo_age"] = dem["classification_1"].reindex(idx).map(num)
    out["demo_c2"]  = dem["classification_2"].reindex(idx).astype(str).replace("nan", np.nan)
    out["demo_c3"]  = dem["classification_3"].reindex(idx).map(num)
    out["demo_c4"]  = dem["classification_4"].reindex(idx).map(lambda s: num(s) if str(s)!="5+" else 5.0)
    out["demo_c5"]  = dem["classification_5"].reindex(idx).map(num)
    out["homeowner"] = dem["homeowner_desc"].reindex(idx).astype(str).replace("nan", np.nan)
    out["kid_cat"]   = dem["kid_category_desc"].reindex(idx).astype(str).replace("nan", np.nan)
    out["has_demo"]  = pd.Series(idx.isin(dem.index).astype(float), index=idx)
    wk = (snapshot_day + 8) // 7
    out["snap_day"]   = float(snapshot_day)
    out["wk_of_year"] = float(wk % 52)
    ang = 2*np.pi*((snapshot_day % 364)/364.0)
    out["ann_sin"] = np.sin(ang); out["ann_cos"] = np.cos(ang)
    return out.reindex(view.households)

new = agent_api.build_features(fn)
print("new feats:", new.shape)
base = agent_api.load_saved("e005_decay_gapcv.parquet")
print("base:", base.shape)
m = base.merge(new.reset_index().rename(columns={"index":"household_key"}), on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape, "| cols:", len(m.columns))
print("dup check:", m.duplicated(["household_key","snapshot_day"]).sum())
path = agent_api.save_table(m, "e010_lifecycle.parquet")
print("saved:", path)
