import agent_api as api, numpy as np, pandas as pd

saved = api.load_saved("e013_storeprod.parquet")
tt = api.train_targets()
y = tt.future_spend_4w.values
print("target: mean %.1f med %.1f std %.1f zero-share %.3f" % (y.mean(), np.median(y), y.std(), (y==0).mean()))
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median",lambda x:(x==0).mean()]))

# ridge proxy on E013 numeric features
df = saved.merge(tt, on=["household_key","snapshot_day"])
num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number]).columns.tolist()
print("n numeric feats:", len(num))
X = df[num].values.astype(float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
mu, sd = X[df.snapshot_day<=431].mean(0), X[df.snapshot_day<=431].std(0)+1e-9
Xs = (X-mu)/sd
tr = (df.snapshot_day<=431).values; va = ~tr
def ridge(lam):
    A = Xs[tr]; b = y[tr]
    I = np.eye(A.shape[1]); I[0,0]=0
    w = np.linalg.solve(A.T@A + lam*I, A.T@b)
    pred = Xs@w
    return pred
best=None
for lam in [1,10,100,300,1000,3000]:
    p = ridge(lam)
    mae = np.abs(p[va]-y[va]).mean()
    if best is None or mae<best[1]: best=(lam,mae)
print("ridge proxy best:", best)
p = ridge(best[0])
res = y-p
print("resid: mean %.2f std %.2f | corr resid vs lag1 %.3f vs te2_ewm %.3f vs te_hh_mean %.3f" % (
    res.mean(), res.std(), np.corrcoef(res, df.spend_4w_lag1)[0,1], np.corrcoef(res, df.te2_ewm)[0,1], np.corrcoef(res, df.te_hh_mean)[0,1]))
# small model: only top outcome features
top = ["spend_4w_lag1","spend_4w_lag2","spend_4w_lag3","te2_mean","te2_ewm","te2_med","te_hh_shrunk","spend_4w_recent","spend_8w","te2_slope","te2_std","te2_zero"]
Xt = df[top].values.astype(float); Xt=np.nan_to_num(Xt,nan=0.0)
mu2,sd2 = Xt[tr].mean(0), Xt[tr].std(0)+1e-9
Xt2=(Xt-mu2)/sd2
for lam in [1,10,100,300,1000]:
    A=Xt2[tr]; I=np.eye(A.shape[1]); I[0,0]=0
    w=np.linalg.solve(A.T@A+lam*I, A.T@y[tr]); pr=Xt2@w
    print("small lam",lam,"val MAE %.3f"%np.abs(pr[va]-y[va]).mean())
# single features
for c in ["spend_4w_lag1","te2_ewm","te2_mean","te_hh_shrunk","spend_4w_recent"]:
    v = df[c].fillna(df[c].median()).values
    print("single",c,"val MAE %.2f"%np.abs(v[va]-y[va]).mean())
