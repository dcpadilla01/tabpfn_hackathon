import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner")
y = tr.future_spend_4w.values.astype(float)
# all-NaN / constant columns in base (train rows)
numcols = [c for c in tr.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(tr[c])]
dead = [c for c in numcols if tr[c].notna().sum()==0 or tr[c].nunique(dropna=True)<=1]
print("dead numeric cols:", dead)
# cheap adds: nf_seasonal (2), coupon_disc_84/match_disc_84 (from e019_true_union), nratio_l/nspend_l/tenure (e017_disc_seasonal)
ns = A.load_saved("nf_seasonal.parquet")[["household_key","snapshot_day","nseas_uplift","nseas_uplift_mean"]]
tu = A.load_saved("e019_true_union.parquet")[["household_key","snapshot_day","coupon_disc_84","match_disc_84"]]
ds = A.load_saved("e017_disc_seasonal.parquet")[["household_key","snapshot_day","nratio_l","nspend_l","tenure"]]
m = tr.merge(ns, on=["household_key","snapshot_day"]).merge(tu, on=["household_key","snapshot_day"]).merge(ds, on=["household_key","snapshot_day"])
for c in ["nseas_uplift","nseas_uplift_mean","coupon_disc_84","match_disc_84","nratio_l","nspend_l","tenure"]:
    x = m[c].values.astype(float); ok = np.isfinite(x)
    print(c, "nan%", round(100*(1-ok.mean()),1), "corr", round(np.corrcoef(x[ok],y[ok])[0,1],4) if ok.sum()>100 else "NA")
# also check spend_ly4w / peer cols corr for context
for c in ["spend_ly4w","peer_p90","zero_frac_full","inactive_run84","n_zero_l13"]:
    x = tr[c].values.astype(float); ok=np.isfinite(x)
    print(c, "nan%", round(100*(1-ok.mean()),1), "corr", round(np.corrcoef(x[ok],y[ok])[0,1],4))