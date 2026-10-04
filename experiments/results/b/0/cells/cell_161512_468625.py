import agent_api as A
import pandas as pd, numpy as np

e011 = A.load_saved("e011_price.parquet")
tt = A.train_targets()
df = e011.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged rows:", len(df), "| dup rows:", df.duplicated(["household_key","snapshot_day"]).sum())

d431 = df[df.snapshot_day==431]
print("day431 rows:", len(d431), "mean y:", d431.future_spend_4w.mean().round(1))
print("spend112 corr with y on 431:", np.corrcoef(pd.to_numeric(d431.spend112,errors="coerce").fillna(0), d431.future_spend_4w)[0,1].round(3))

# minimal ridge: single feature spend112
def ridge_eval(cols, fit_days, eval_days, lam=3.0):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-10,10).fillna(0).values
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values
    b = np.linalg.solve(Xs[tr].T@Xs[tr]+lam*np.eye(len(cols)), Xs[tr].T@y[tr])
    p = Xs@b
    m = ~tr
    return p[m], y[m]

p, y = ridge_eval(["spend112"], [95,123,151,179,207,235,263,291,319,347,375,403], [431])
print("single-feat ridge: MAE=%.2f  mean pred=%.1f mean y=%.1f  pred std=%.1f" % (np.abs(p-y).mean(), p.mean(), y.mean(), p.std()))

# now check Xs magnitudes for core64
feats = [c for c in e011.columns if c not in ("household_key","snapshot_day")]
X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
sd = X.std()
print("\nfeatures with sd=0:", (sd==0).sum())
print("features with sd>1000:", (sd>1000).sum())
print(sd.sort_values(ascending=False).head(8).round(0))
# check for features with extreme values pre-clip
mx = X.abs().max()
print("\nmax |x| top:", mx.sort_values(ascending=False).head(8).round(0))
