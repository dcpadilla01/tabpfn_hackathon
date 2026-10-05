import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

TRAIN_DAYS = list(range(95, 432, 28))
ANCHOR_DAYS = TRAIN_DAYS + [459, 487]

def build(view, s):
    tx = view.table("transactions")
    hs = np.asarray(view.households).ravel()
    # ---- household-day spend table (sorted by hh, day) ----
    g = tx.groupby(["household_key","day"], sort=True)["sales_value"].sum().reset_index()
    hh_arr = g["household_key"].values
    day_arr = g["day"].values.astype(np.int64)
    sp_arr = g["sales_value"].values.astype(float)
    uniq = np.unique(hh_arr)
    starts = np.searchsorted(hh_arr, uniq, side="left")
    ends = np.searchsorted(hh_arr, uniq, side="right")
    csg = np.concatenate([[0.0], np.cumsum(sp_arr)])
    h2i = {int(h): i for i, h in enumerate(uniq)}
    # ---- basket-level table (sorted by hh, day) ----
    bk = tx.groupby("basket_id", as_index=False).agg(hh=("household_key","first"),
                                                     day=("day","first"),
                                                     sp=("sales_value","sum"))
    bk = bk.sort_values(["hh","day"], kind="stable")
    bhh = bk["hh"].values; bday = bk["day"].values.astype(np.int64); bsp = bk["sp"].values.astype(float)
    buniq = np.unique(bhh)
    bstarts = np.searchsorted(bhh, buniq, side="left")
    bends = np.searchsorted(bhh, buniq, side="right")
    b2i = {int(h): i for i, h in enumerate(buniq)}
    anchors = [a for a in ANCHOR_DAYS if a + 28 <= s]
    rows = np.zeros((len(hs), 22), dtype=float)
    NEW = ["te2_n","te2_mean","te2_med","te2_std","te2_cv","te2_zero","te2_min","te2_max",
           "te2_slope","te2_ewm","gap_mean_all","gap_med_all","gap_std_all","gap_mean_8w",
           "exp_trips_4w","exp_spend_4w","exp_spend_112","spend_7d","nbask_7d","active_7d",
           "max_basket_4w","nbask_ratio_4_28"]
    for r, h in enumerate(hs):
        i = h2i.get(int(h)); bi = b2i.get(int(h))
        if i is None:
            continue
        a0, a1 = starts[i], ends[i]
        d = day_arr[a0:a1]; fd = d[0]
        # outcome series over eligible anchors
        vals = []
        for anc in anchors:
            if fd <= anc - 84:
                j = np.searchsorted(d, anc, side="right")
                k = np.searchsorted(d, anc+28, side="right")
                vals.append(csg[a0+k] - csg[a0+j])
        n = len(vals)
        if n:
            v = np.array(vals); m = v.mean()
            rows[r,0] = n; rows[r,1] = m; rows[r,2] = np.median(v); rows[r,3] = v.std()
            rows[r,4] = v.std()/m if m > 0 else np.nan
            rows[r,5] = (v==0).mean(); rows[r,6] = v.min(); rows[r,7] = v.max()
            k3 = min(3, n)
            rows[r,8] = v[-k3:].mean() - v[:k3].mean()
            w = 0.7 ** np.arange(n-1, -1, -1)
            rows[r,9] = (v*w).sum()/w.sum()
        if len(d) > 1:
            gp = np.diff(d).astype(float)
            rows[r,10] = gp.mean(); rows[r,11] = np.median(gp); rows[r,12] = gp.std()
        d8 = d[d > s-56]
        if len(d8) > 1: rows[r,13] = np.diff(d8).mean()
        d112 = d[d > s-112]
        gm112 = np.diff(d112).mean() if len(d112) > 1 else np.nan
        # expected trips x basket size
        gm = rows[r,10]
        if bi is not None:
            b0, b1 = bstarts[bi], bends[bi]
            bd = bday[b0:b1]; bs = bsp[b0:b1]
            if np.isfinite(gm) and gm > 0:
                rows[r,14] = 28.0/gm
                rows[r,15] = 28.0/gm * bs.mean()
            if np.isfinite(gm112) and gm112 > 0:
                j = np.searchsorted(bd, s-784, side="right")
                nb112 = b1-b0-j
                if nb112 > 0:
                    sp112 = csg[a0+np.searchsorted(d, s-784, side="right")] - csg[a0] if len(d) else 0.0
                    rows[r,16] = 28.0/gm112 * (sp112/nb112)
            j7 = np.searchsorted(bd, s-7, side="right"); n7 = b1-b0-j7
            rows[r,18] = n7; rows[r,19] = 1.0 if n7 > 0 else 0.0
            j4 = np.searchsorted(bd, s-28, side="right"); n4 = b1-b0-j4
            rows[r,20] = bs[j4:].max() if n4 > 0 else 0.0
            j28 = np.searchsorted(bd, s-196, side="right"); n28 = b1-b0-j28
            if n28 > 0: rows[r,21] = n4/(n28/7.0)
        rows[r,17] = csg[a0+np.searchsorted(d, s-7, side="right")] - csg[a0]
    out = pd.DataFrame(rows, index=pd.Index(hs, name="household_key"), columns=NEW)
    if s == TRAIN_DAYS[0]:
        print("snap", s, "hh", len(hs), "anchors", len(anchors), "te2_n>0 share", round(float((out["te2_n"]>0).mean()),3))
    return out

feats = A.build_features(build)
print("feats:", feats.shape)
e7 = A.load_saved("e007_te.parquet")
full = e7.merge(feats, on=["household_key","snapshot_day"], how="inner")
print("full:", full.shape)
assert len(full) == 36426
tt = A.train_targets()
dg = tt.merge(full, on=["household_key","snapshot_day"], how="left")
y = dg[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))
med = float(np.median(y))
lag1 = dg["spend_4w_lag1"].fillna(med).values
print("corr te2_mean~y:", round(float(np.corrcoef(dg["te2_mean"].fillna(0), y)[0,1]),3),
      " corr te2_mean~te_hh_mean:", round(float(np.corrcoef(dg["te2_mean"].fillna(0), dg["te_hh_mean"].fillna(0))[0,1]),3))
print("single MAE te2_mean chain:", round(mae(dg["te2_mean"].fillna(dg["spend_4w_lag1"]).fillna(med).values),2),
      " te_hh_mean chain:", round(mae(dg["te_hh_mean"].fillna(dg["spend_4w_lag1"]).fillna(med).values),2))
print("te2_zero NaNs:", int(dg["te2_zero"].isna().sum()), " exp_spend_4w corr:",
      round(float(np.corrcoef(dg["exp_spend_4w"].fillna(0), y)[0,1]),3))
path = A.save_table(full, "e012_outcome2.parquet")
print("saved:", path)
