import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
e1 = A.load_saved("e001_history.parquet")
snap = A.snapshot(459)
tx = snap.transactions
first_day = tx.groupby("household_key").day.min()
g = tx.assign(wk=tx.day//7).groupby("wk").agg(sales=("sales_value","sum"), hh=("household_key","nunique"))
macro_w = (g.sales/g.hh); macro_w = macro_w[macro_w.index <= 64]
train_days = A.snapshot_days()["train"]

rows = {}
for s in train_days:
    t = tx[tx.day <= s]
    wi = ((s - t.day) // 28).clip(lower=0)
    piv = t.assign(wi=wi).pivot_table(index="household_key", columns="wi", values="sales_value", aggfunc="sum").reindex(columns=range(26)).fillna(0.0)
    fd = first_day.reindex(piv.index).values
    hh_idx = piv.index
    # year-ago future window: days s-363 .. s-336
    flo, fhi = s-363, s-336
    tw = t[(t.day >= flo) & (t.day <= fhi)].groupby("household_key").sales_value.sum()
    mfly = tw.reindex(hh_idx).fillna(0.0).values
    obs_lo = np.maximum(flo, fd); obs_hi = np.minimum(fhi, s)
    frac = np.clip((obs_hi - obs_lo + 1) / 28.0, 0, 1)  # observable fraction of the 28d window
    mfly_adj = np.where(frac > 0.25, mfly / np.maximum(frac, 0.25), np.nan)
    # macro drift
    mw_t = macro_w[macro_w.index <= s//7]
    macro_lvl = mw_t.tail(8).mean(); macro_all = mw_t.mean()
    # year-ago future weeks macro (seasonality) - only where available
    wf0, wf1 = (s+1)//7, (s+28)//7
    ya = macro_w.reindex(range(wf0-52, wf1-52+1)).dropna()
    mfs = (ya.mean()/macro_lvl) if len(ya) >= 2 else np.nan
    df = pd.DataFrame({"household_key": hh_idx, "mfly": mfly, "mfly_frac": frac,
                       "mfly_adj": mfly_adj, "macro_rel": macro_lvl/macro_all,
                       "macro_fut_seas": mfs, "tenure": s - fd})
    df["mfut_rel"] = df.mfly_adj / np.maximum(piv[0].values, 1.0)
    df["mfly_raw_rel"] = df.mfly / np.maximum(piv[0].values, 1.0)
    df["snapshot_day"] = s
    rows[s] = df
C = pd.concat(rows.values(), ignore_index=True)

base = tt.merge(e8, on=["household_key","snapshot_day"])
M = base.merge(C, on=["household_key","snapshot_day"], how="left")
M["h200"] = np.maximum(M.e6-200, 0); M["h400"] = np.maximum(M.e6-400, 0)
M["rank_e6"] = M.groupby("snapshot_day").e6.rank(pct=True)
M["rank_usual13"] = M.groupby("snapshot_day").usual13.rank(pct=True)
M["sqrt_e6"] = np.sqrt(M.e6.clip(lower=0))
E1raw = ["spend_7","spend_56","spend_84","spend_182","spend_365","nb_28","nb_56","nb_all","avg_basket_28",
         "days_since_last","spend_prev28","trend28","qty_28","nprod_28","spend_ly28","weekly_rate_84"]
M = M.merge(e1[["household_key","snapshot_day"]+E1raw], on=["household_key","snapshot_day"], how="left")

# correlations of new macro feats with E008 ly features
for a,b in [("mfly","ly_spend"),("mfly_adj","ly_spend"),("mfut_rel","ly_ratio"),("mfly_raw_rel","ly_ratio")]:
    ok = M[a].notna() & M[b].notna()
    print(f"corr({a},{b}) = {np.corrcoef(M.loc[ok,a], M.loc[ok,b])[0,1]:.3f}  (n={ok.sum()})")
print("mfly_adj NaN frac by snapshot:"); print(M.groupby("snapshot_day").mfly_adj.apply(lambda x: x.isna().mean().round(2)).to_string())

def loso(df, feats, hold_last=1, lam=10.0):
    days = sorted(df.snapshot_day.unique())
    ho = days[-hold_last:]; tr = [d for d in days if d not in ho]
    Xtr = df[df.snapshot_day.isin(tr)][feats].apply(pd.to_numeric, errors="coerce")
    ytr = df[df.snapshot_day.isin(tr)].future_spend_4w.values
    Xho = df[df.snapshot_day.isin(ho)][feats].apply(pd.to_numeric, errors="coerce")
    mu, sd = Xtr.mean(), Xtr.std()+1e-9
    Xtr_s = ((Xtr-mu)/sd).fillna(0).values; Xho_s = ((Xho-mu)/sd).fillna(0).values
    b = np.linalg.solve(Xtr_s.T@Xtr_s + lam*np.eye(len(feats)), Xtr_s.T@ytr)
    return np.abs(Xho_s@b - df[df.snapshot_day.isin(ho)].future_spend_4w.values).mean()

E8 = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
b1 = loso(M, E8); b2 = loso(M, E8, hold_last=2)
print(f"\nBASE E008: {b1:.3f} / {b2:.3f}")
tests = {
 "mfly_raw": ["mfly"],
 "mfly_adj+frac": ["mfly_adj","mfly_frac"],
 "mfut_rel": ["mfut_rel"],
 "mfly_raw_rel": ["mfly_raw_rel"],
 "macro_rel": ["macro_rel"],
 "macro_fut_seas": ["macro_fut_seas"],
 "tenure": ["tenure"],
 "macro_grp": ["mfly_adj","mfly_frac","mfut_rel","macro_rel"],
 "macro_grp+ten": ["mfly_adj","mfly_frac","mfut_rel","macro_rel","tenure"],
 "hinges": ["h200","h400"],
 "sqrt": ["sqrt_e6"],
 "ranks": ["rank_e6","rank_usual13"],
 "w0w5": ["mfly"],  # placeholder replaced below
}
# raw windows w0..w5
M["w0"]=M.spend_28  # spend_28 == w0
# compute w1..w5 quickly from e1? not present; approximate via spend_56-spend_28 etc.
M["w1"] = M.spend_56 - M.spend_28
M["w2"] = M.spend_84 - M.spend_56
M["w3"] = M.spend_182 - M.spend_84  # 98d window, not exact; rough
tests["w12"] = ["w1","w2"]
tests["e1union"] = E1raw
tests["macro+hinges"] = ["mfly_adj","mfly_frac","mfut_rel","macro_rel","h200","h400"]
for k,v in tests.items():
    if k=="w0w5": continue
    print(f"{k:16s}: {loso(M, E8+v):.3f} / {loso(M, E8+v, hold_last=2):.3f}")