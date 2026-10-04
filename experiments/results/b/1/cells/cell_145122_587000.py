import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()
CATS = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]

def fit_pred(tr_max=403, te=431, alpha=100, base=None):
    df = (base if base is not None else e1).merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[CATS].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    A = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A[0,0]-=alpha
    w = np.linalg.solve(A, Xtr.T@y[tr])
    pred = np.c_[np.ones(te_m.sum()), Xs[te_m]]@w
    return pd.DataFrame({"y":y[te_m],"pred":pred})

p = fit_pred()
p["abs_err"] = (p.y-p.pred).abs()
p["bucket"] = pd.qcut(p.y, [0,.25,.5,.75,.9,.95,1], duplicates="drop")
print("MAE decomposition by true-spend bucket (day 431):")
print(p.groupby("bucket", observed=True).agg(n=("y","size"), mae=("abs_err","mean"), mean_y=("y","mean"), mean_pred=("pred","mean"), bias=("pred","mean")).round(1))
print("\ntotal MAE:", round(p.abs_err.mean(),2))
# how much of MAE comes from top 10% of PREDICTED spend?
p["pbucket"] = pd.qcut(p.pred, [0,.5,.9,.95,1], duplicates="drop")
print(p.groupby("pbucket", observed=True).agg(n=("y","size"), mae=("abs_err","mean")).round(1))
# zero-true households
z = p[p.y==0]
print("\ntrue-zero households: n=%d, MAE=%.1f, mean pred=%.1f -> share of total MAE: %.1f%%" %
      (len(z), z.abs_err.mean(), z.pred.mean(), 100*z.abs_err.sum()/p.abs_err.sum()))
# clip predictions at 0?
print("MAE with pred clipped at 0:", round((p.y-p.pred.clip(lower=0)).abs().mean(),2))
# what does optimal global scaling do?
from itertools import product
best=None
for c in [0.8,0.9,1.0,1.1]:
    m = (p.y-c*p.pred).abs().mean()
    if best is None or m<best[1]: best=(c,m)
print("best global scale on pred:", best)
# per-snapshot-day bias in train: is there a level shift?
print("\ntrain mean target by day (from earlier): means rise 130->146; check pred bias by day")
for d in [403, 431]:
    q = fit_pred(tr_max=d-28, te=d)
    print(d, "MAE:", round((q.y-q.pred).abs().mean(),2), "mean y:", round(q.y.mean(),1), "mean pred:", round(q.pred.mean(),1))