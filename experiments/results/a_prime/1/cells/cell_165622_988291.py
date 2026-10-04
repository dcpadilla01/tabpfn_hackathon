import numpy as np, pandas as pd

tt = train_targets()
def loso(tab, label, alpha=50.0):
    d = tab.merge(tt, on=["household_key","snapshot_day"])
    num = d.select_dtypes(include=[np.number])
    cats = [c for c in d.columns if d[c].dtype==object or str(d[c].dtype)=="category"]
    X = num.drop(columns=["future_spend_4w"], errors="ignore")
    for c in cats:
        oh = pd.get_dummies(d[c].astype("category"), prefix=c[:6], dummy_na=True)
        X = pd.concat([X, oh.astype(float)], axis=1)
    keep = X.columns[(X.nunique(dropna=True)>1).values]
    X = X[keep]
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = d["future_spend_4w"].values.astype(float)
    days = d["snapshot_day"].values
    mu, sd = X.mean().values, X.std().values+1e-9
    Xs = np.clip((X.values-mu)/sd, -5, 5)
    errs=[]
    for s in sorted(set(days))[:13]:
        tr = days!=s; te = days==s
        Xt, yt = Xs[tr], y[tr]
        A = Xt.T@Xt + alpha*np.eye(Xt.shape[1])
        w = np.linalg.solve(A, Xt.T@yt)
        pred = np.clip(Xs[te]@w, 0, 1500)
        errs.append(np.abs(pred-y[te]).mean())
    return np.mean(errs)

a = load_saved("e019_everything.parquet")
m = load_saved("e018_union_full.parquet")
print("e019:", loso(a,"x"))
print("union:", loso(m,"x"))

# What if we add ONLY a few timing features that don't exist in e019?
b = load_saved("e018_timing_hazard.parquet")
cand = ["overdue_days","overdue_ratio","next_trip_eta","exp_trips28","gap_last5_mean","rhythm_break","burst7","dow_sd","m_spend28_mean","m_spend28_med","zero_frac_l13","zero_frac_l26","act_w4","act_w8","act_w13","trend14_42"]
cand = [c for c in cand if c not in a.columns]
small = a.merge(b[["household_key","snapshot_day"]+cand], on=["household_key","snapshot_day"])
print("small timing add:", loso(small,"x"))
# try even smaller: 6
cand6 = ["overdue_days","next_trip_eta","gap_last5_mean","burst7","m_spend28_mean","zero_frac_l13"]
cand6 = [c for c in cand6 if c not in a.columns]
small6 = a.merge(b[["household_key","snapshot_day"]+cand6], on=["household_key","snapshot_day"])
print("6 timing add:", loso(small6,"x"))