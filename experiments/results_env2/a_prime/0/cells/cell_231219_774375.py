import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values

# correlation of target with candidate predictors (on train rows)
cands = ["te_hh_shrunk","te_hh_mean","spend_4w","spend_8w","spend_12w","spend_28w","spend_56w","spend_112w",
         "spend_4w_lag1","spend_4w_lag2","spend_4w_lag3","spend_112_mean4","spend_112_std","spend_112_cv",
         "nbask_4w","nbask_8w","nbask_112","days_since_last","tenure_days","avg_basket_12w",
         "spend_per_basket_112","trend_4_8","trend_4_28","gap_mean_112","gap_max_112","gap_std_112",
         "n_campaign_targets","days_since_last_tgt_start","share_disp_28","share_mail_28","te_prior","te_bin"]
rows=[]
for c in cands:
    if c in df:
        x = df[c].astype(float)
        m = x.notna() & ~np.isnan(y)
        if m.sum()>100:
            rows.append((c, round(float(np.corrcoef(x[m], y[m])[0,1]),3)))
rows.sort(key=lambda r:-abs(r[1]))
for r in rows: print(r)
