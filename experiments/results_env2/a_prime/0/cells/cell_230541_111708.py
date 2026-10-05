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
# new: zero-lag count among available, zero share of 4w windows (last 6 lags), churn-adjusted forecast
Z = S[["lag1","lag2","lag3","lag4","lag5","lag6"]]
tr["oh_zero_share6"] = (Z==0).sum(1) / Z.notna().sum(1).clip(lower=1)
tr["oh_zero_any6"] = ((Z==0).sum(1) > 0).astype(float)
tr["oh_ewm15_x_active"] = tr["oh_ewm1.5"] * (1 - tr["oh_zero_share6"].fillna(0))
tr["oh_ewm15_x_zeroflag"] = tr["oh_ewm1.5"] * (1 - tr["oh_zero_any6"])
tr["oh_min3"] = S[["lag1","lag2","lag3"]].min(1)
tr["oh_max3"] = S[["lag1","lag2","lag3"]].max(1)
tr["oh_trend3"] = S["lag1"] - S["lag3"]
tr["oh_cv3"] = S[["lag1","lag2","lag3"]].std(1) / S[["lag1","lag2","lag3"]].mean(1).replace(0, np.nan)
new_cols = ["oh_zero_share6","oh_zero_any6","oh_ewm15_x_active","oh_ewm15_x_zeroflag","oh_min3","oh_max3","oh_trend3","oh_cv3"]
m2 = m.merge(tr[["household_key","snapshot_day","oh_ewm1.5","oh_ewm4","oh_n"]+new_cols], on=["household_key","snapshot_day"], how="left")

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
        "spend_4w_lag1","spend_4w_lag2","spend_4w_lag3","spend_112_mean4","spend_112_min4","spend_112_max4",
        "spend_112_std","spend_112_cv","gap_mean_112","gap_max_112","gap_std_112","nweek_active_112",
        "week_active_share_112","spend_per_week_112","spend_per_active_week_112","spend_per_basket_112",
        "spend_per_basket_total","spend_per_day_total","avg_price_line_112","avg_units_line_112",
        "disc_share_112","neg_spend_share_112","nprod_112","nstore_112","mean_trans_time_112"]
oh = ["oh_ewm1.5","oh_ewm4","oh_n"]
baseE = te+core+["snapshot_day"]+oh
print("baseE:", round(ridge_eval(baseE,403,431),3), round(ridge_eval(baseE,375,403),3))
for c in new_cols:
    v = baseE+[c]
    print(f"+{c:22s} @431 {ridge_eval(v,403,431):7.3f}  @403 {ridge_eval(v,375,403):7.3f}")
allnew = baseE+new_cols
print("baseE+allnew:", round(ridge_eval(allnew,403,431),3), round(ridge_eval(allnew,375,403),3), round(ridge_eval(allnew,347,375),3))
print("baseE @375:", round(ridge_eval(baseE,347,375),3))
# also standalone quality of new cols
def mae(p,t): return np.mean(np.abs(np.asarray(p)-np.asarray(t)))
t431 = m2[m2.snapshot_day==431]
for c in new_cols+["oh_ewm1.5"]:
    print("standalone", c, round(mae(t431[c].fillna(t431[c].median()).values, t431.future_spend_4w.values),3))
