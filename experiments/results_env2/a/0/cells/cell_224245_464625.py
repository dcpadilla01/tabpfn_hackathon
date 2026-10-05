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
pr["err"] = (pr.pred - pr.future_spend_4w).abs()

print("=== error structure at 431 ===")
print("overall MAE:", round(pr.err.mean(),2))
print("\nby target bucket:")
pr["tb"] = pd.cut(pr.future_spend_4w, [-1,0.01,50,100,200,400,10000])
print(pr.groupby("tb").agg(n=("err","size"), mae=("err","mean"), med_pred=("pred","median")))
print("\nby days_since_last bucket:")
pr["dsl"] = pd.cut(pr.days_since_last, [-1,7,14,28,56,10000])
print(pr.groupby("dsl").agg(n=("err","size"), mae=("err","mean"), med_target=("future_spend_4w","median"), med_pred=("pred","median")))
print("\nby tenure bucket:")
pr["tn"] = pd.cut(pr.tenure, [-1,150,300,500,10000])
print(pr.groupby("tn").agg(n=("err","size"), mae=("err","mean")))
print("\nzero-target rows: n=%d, mean pred=%.1f" % ((pr.future_spend_4w==0).sum(), pr.loc[pr.future_spend_4w==0,"pred"].mean()))
print("zero-target MAE contribution: %.1f of total %.1f" % (pr.loc[pr.future_spend_4w==0,"err"].mean()* (pr.future_spend_4w==0).mean(), pr.err.mean()))
imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\ntop-15 importance:"); print(imp.head(15).round(4))
