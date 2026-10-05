import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
df = agent_api.load_saved("e007_te.parquet")
tt = agent_api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
tr = m[["household_key","snapshot_day","future_spend_4w"]].sort_values(["household_key","snapshot_day"])
g = tr.groupby("household_key")["future_spend_4w"]
lagcols = [f"lag{k}" for k in range(1,14)]
S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
for hl in [1.5, 4]:
    w = 0.5**((np.arange(1,14)-1)/hl)
    M = S.notna().values.astype(float) * w[None,:]
    tr[f"oh_ewm{hl}"] = (S.fillna(0).values * (M/ M.sum(1,keepdims=True))).sum(1)
tr["oh_n"] = g.cumcount()
m2 = m.merge(tr[["household_key","snapshot_day","oh_ewm1.5","oh_ewm4","oh_n"]], on=["household_key","snapshot_day"], how="left")

# raw demographics (static) — merge on household
demo = agent_api.snapshot().demographics
print("demo rows:", len(demo), demo.columns.tolist())
d2 = demo.copy()
for c in d2.columns:
    if c != "household_key": d2[c] = d2[c].astype(str)
m2 = m2.merge(d2, on="household_key", how="left")
m2["has_demo"] = m2["household_key"].isin(set(demo.household_key)).astype(float)

def prep(mm, cols):
    X = mm[cols].copy()
    for c in cols:
        if X[c].dtype == object: X[c] = pd.factorize(X[c])[0]
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
def ridge_eval(cols, fit_max, eval_day, lam=10.0):
    mm = m2[m2.snapshot_day<=eval_day]
    trm = (mm.snapshot_day<=fit_max).values; tem = (mm.snapshot_day==eval_day).values
    Xa = prep(mm, cols); ya = mm.future_spend_4w.values
    mu, sd = Xa[trm].mean(0), Xa[trm].std(0)+1e-9
    A = np.c_[np.ones(trm.sum()), (Xa[trm]-mu)/sd]; B = np.c_[np.ones(tem.sum()), (Xa[tem]-mu)/sd]
    I = np.eye(A.shape[1]); I[0,0]=0
    beta = np.linalg.solve(A.T@A+lam*I, A.T@ya[trm])
    return np.mean(np.abs(B@beta - ya[tem]))

te = ["te_hh_mean","te_hh_n","te_hh_shrunk","te_bin","te_prior","te_size","te_home","te_kid"]
core = ["spend_4w","nbask_4w","spend_8w","nbask_8w","spend_12w","nbask_12w","spend_28w","nbask_28w",
        "spend_56w","nbask_56w","spend_112w","nbask_112w","spend_total","nbask_total","tenure_days",
        "days_since_last","avg_basket_12w","trend_4_8","trend_4_28","active_4w",
        "spend_4w_lag1","spend_4w_lag2","spend_4w_lag3","spend_4w_lag4","spend_4w_lag5","spend_4w_lag6",
        "spend_112_mean4","spend_112_min4","spend_112_max4","spend_112_std","spend_112_cv",
        "gap_mean_112","gap_max_112","gap_std_112","nweek_active_112","week_active_share_112",
        "spend_per_week_112","spend_per_active_week_112","spend_per_basket_112","spend_per_basket_total",
        "spend_per_day_total","avg_price_line_112","avg_units_line_112","disc_share_112","neg_spend_share_112",
        "nprod_112","nstore_112","mean_trans_time_112"]
dsh = [c for c in df.columns if c.startswith("dsh_") and c!="dsh_ "] + ["bsh_National","bsh_Private"]
mkt = ["n_campaign_targets","days_since_last_tgt_start","tgt_active_now","tgt_active_future4w","tgt_TypeA","tgt_TypeB","tgt_TypeC",
       "n_redemptions_total","days_since_last_redemption","n_redemptions_28d","n_redemptions_56d","n_redemptions_112d"]
disp = ["share_disp_28","share_mail_28","share_lines_disp_28","share_baskets_disp_28","share_disp_112","share_mail_112",
        "share_lines_disp_112","share_baskets_disp_112","share_spend_nowdisp_28"]
demo_cols = [c for c in d2.columns if c!="household_key"] + ["has_demo"]
cal = ["snapshot_day"]
oh = ["oh_ewm1.5","oh_ewm4","oh_n"]
print("sizes: te%d core%d dsh%d mkt%d disp%d demo%d"%(len(te),len(core),len(dsh),len(mkt),len(disp),len(demo_cols)))

variants = {
 "A core+te+demo+cal+oh": te+core+demo_cols+cal+oh,
 "B A+dsh": te+core+demo_cols+cal+oh+dsh,
 "C B+mkt": te+core+demo_cols+cal+oh+dsh+mkt,
 "D C+disp(=E007lean)": te+core+demo_cols+cal+oh+dsh+mkt+disp,
 "E core+te+cal+oh (no demo)": te+core+cal+oh,
 "F E+mkt+disp": te+core+cal+oh+mkt+disp,
}
for name, cols in variants.items():
    e431 = ridge_eval(cols,403,431); e403 = ridge_eval(cols,375,403)
    print(f"{name:30s} n={len(cols):3d} @431 {e431:7.3f}  @403 {e403:7.3f}")
print("E007 full + oh + demo:", round(ridge_eval([c for c in df.columns if c not in ('household_key','snapshot_day','day','week','tenure','snap_day','snap_week','index')]+oh+demo_cols,403,431),3))
