import agent_api as A
import pandas as pd, numpy as np

# --- build new features through build_features (sees only data <= snapshot day) ---
def fn(view, snapshot_day):
    hh = pd.Index(view.households)
    t = view.transactions
    out = pd.DataFrame(index=hh)
    # 1) calendar-aligned last-year forward window: [D-363, D-336] mirrors target [D+1, D+28] LY
    lo, hi = snapshot_day-363, snapshot_day-336
    m = (t.day >= lo) & (t.day <= hi)
    sp = t.loc[m].groupby("household_key").sales_value.sum()
    first = t.groupby("household_key").day.min()
    out["fwd_ly"] = sp.reindex(hh).fillna(0.0)
    out["fwd_ly_valid"] = (first.reindex(hh).fillna(10**9) <= lo)
    # 2) empirical 4-week activity rates from weekly rolling sums
    wk = t.groupby(["household_key","week_no"]).sales_value.sum().unstack(fill_value=0.0)
    wk = wk.reindex(index=hh, fill_value=0.0)
    w_end = int((snapshot_day + 8)//7)
    allw = np.arange(1, w_end+1)
    wk = wk.reindex(columns=allw, fill_value=0.0).fillna(0.0)
    R = wk.rolling(4, axis=1, min_periods=4).sum()   # 4-week spend ending at week w
    lo_w = max(4, w_end-51)
    vals = R.loc[:, lo_w:w_end].values
    with np.errstate(invalid="ignore"):
        out["act_rate_4w"] = np.nanmean((vals > 0).astype(float), axis=1)
    lo_w2 = max(4, w_end-15)
    vals2 = R.loc[:, lo_w2:w_end].values
    with np.errstate(invalid="ignore"):
        out["act_rate_16w"] = np.nanmean((vals2 > 0).astype(float), axis=1)
    return out

newf = A.build_features(fn)
print("newfeat:", newf.shape)
print(newf.drop(columns=["household_key","snapshot_day"]).describe().round(3))
A.save_table(newf, "newfeat.parquet")

# --- inert feature audit on E011 ---
e011 = A.load_saved("e011_price.parquet")
tt = A.train_targets()
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")
X = df[feats].apply(pd.to_numeric, errors="coerce")
sd = X.std()
inert = list(sd[sd==0].index)
print("\ninert (sd=0 on train):", len(inert), inert[:40])

# --- probe: newfeat added to core+lag+comp ---
def huber_eval(df, cols, fit_days, eval_days, lam=1.0, delta=30.0, n_iter=25):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = np.column_stack([np.ones(len(d)), ((X-mu)/sd).clip(-10,10).fillna(0).values])
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values; m = ~tr
    Xtr, ytr = Xs[tr], y[tr]
    b = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    for _ in range(n_iter):
        r = ytr - Xtr@b
        w = np.minimum(1.0, delta/np.maximum(np.abs(r),1e-9))
        Xw = Xtr * w[:,None]
        b = np.linalg.solve(Xtr.T@Xw + lam*np.eye(Xtr.shape[1]), Xw.T@ytr)
    return np.abs(Xs[m]@b - y[m]).mean()

tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
df2 = df.merge(newf.drop(columns=["household_key","snapshot_day"]), left_index=True, right_index=True) if False else df
# merge on keys properly:
df2 = df.merge(newf, on=["household_key","snapshot_day"], how="left")
base = [c for c in feats if c not in inert]  # all E011 features minus inert
def grp(pre): return [c for c in feats if c.startswith(pre)]
dsp = grp("dsp_"); ap = grp("ap_"); pt = grp("price_trend"); qty = grp("qty_"); sp = grp("spend_p")
iv = grp("iv_")+grp("bk_")+["n_gaps112"]
lag = grp("lag_")+["dec_spend14","dec_spend7","dec_trips","gap_mean","gap_std","gap_med","ntrip112"]
mkt = grp("tA_")+grp("tB_")+grp("tC_")+grp("red_")+["n_tgt_ever","n_tgt_active","targeted_flag","days_since_first_tgt","days_since_last_tgt","tgt_active_remaining","cpn_prod_spend28","cpn_prod_share28"]
demo = grp("demo_")+["has_demo"]
comp = ["ndept28","spend_w0","spend_w1","spend_w2","spend_w3","share_w0","rwspend84","active_weeks112","weekend_share28","zero28","ratio28_364","toptrip_share28","spend28_log","spend56_log","spend112_log","lt_spend_log","nbaskets28"]
dm = [c for c in feats if c.startswith("dm_")]
misc = ["pl_share","disc_share2","c_qty112","macro_ratio","mom_28_112","mom_112_364","upt28","upt112","maxtrip_qty28","share_qty_toptrip"]
clp = core_lag_comp = [c for c in feats if c not in dsp+ap+pt+qty+sp+iv+lag+mkt+demo+comp+dm+misc]
nf = ["fwd_ly","fwd_ly_valid","act_rate_4w","act_rate_16w"]
df2["exp_spend"] = df2["act_rate_4w"] * pd.to_numeric(df2["spend112"],errors="coerce").fillna(0)/4.0
df2["exp_spend16"] = df2["act_rate_16w"] * pd.to_numeric(df2["spend112"],errors="coerce").fillna(0)/4.0
df2["fwd_ly_blend"] = df2["fwd_ly"].fillna(0)*0.5 + pd.to_numeric(df2["spend28"],errors="coerce").fillna(0)*0.5

print("\nPROBE (huber delta=30; fit<=403 eval 431 | fit<=375 eval 403):")
for name, cols in [("prune142", clp+lag+comp+mkt+dm+demo),
                   ("prune116", clp+lag+comp),
                   ("prune116+nf", clp+lag+comp+nf),
                   ("prune116+nf+exp", clp+lag+comp+nf+["exp_spend","exp_spend16","fwd_ly_blend"])]:
    m431 = huber_eval(df2, cols, tr_days, [431]); m403 = huber_eval(df2, cols, tr_days[:-1], [403])
    print(f"  {name:20s} n={len(cols):3d}  431={m431:7.3f}  403={m403:7.3f}")
# individual nf deltas on prune116
b0 = huber_eval(df2, clp+lag+comp, tr_days, [431])
for c in nf+["exp_spend","exp_spend16","fwd_ly_blend"]:
    print(f"  +{c:14s} {huber_eval(df2, clp+lag+comp+[c], tr_days, [431]):.3f} (d={huber_eval(df2, clp+lag+comp+[c], tr_days, [431])-b0:+.3f})")
