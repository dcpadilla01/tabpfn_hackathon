import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
BASE = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
        objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")

td = [d for d in range(95,432,28) if d < 431]
tr = df[df.snapshot_day.isin(td)]
m = xgb.XGBRegressor(**BASE).fit(tr[FEATS], tr.future_spend_4w)
pr = df[df.snapshot_day==431].copy()
pr["pred"] = m.predict(pr[FEATS])

# churn definition: no purchase in trailing 28d at snapshot (days_since_last > 28)
pr["churn28"] = pr.days_since_last > 28
print("churn28 rows: %d (%.1f%%)" % (pr.churn28.sum(), 100*pr.churn28.mean()))
print("target==0 rate by churn28:")
print(pr.groupby("churn28").future_spend_4w.apply(lambda s: (s==0).mean()).round(3))
print("MAE by churn28:")
print(pr.groupby("churn28").apply(lambda g: (g.pred-g.future_spend_4w).abs().mean()).round(2))

# does model already use dsl? check partial dependence-ish: pred vs dsl among churn28
print("\nchurn28: pred distribution:", pr.loc[pr.churn28,"pred"].describe().round(1).to_dict())

# what about households active recently but low spend: how often do they go to 0?
pr["dsl_b"] = pd.cut(pr.days_since_last, [-1,7,14,21,28,42,56,10000])
print("\nzero-rate & mean target by dsl bucket:")
print(pr.groupby("dsl_b").agg(n=("pred","size"), zero_rate=("future_spend_4w", lambda s:(s==0).mean()), mean_t=("future_spend_4w","mean"), med_pred=("pred","median")).round(2))

# check wk_cv8 / gap features for churn signal
print("\ncorr of features with (future==0) among dsl<=28 rows:")
sub = pr[pr.days_since_last<=28]
for c in ["wk_cv8","gap_max_84","gap_mean_84","active_weeks8","trips_28","spend_28","exp4w_blend","wk_std8","basket_max_84"]:
    print(f"  {c:16s} {np.corrcoef(sub[c], (sub.future_spend_4w==0).astype(float))[0,1]:.3f}")
