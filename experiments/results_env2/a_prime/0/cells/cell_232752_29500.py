import agent_api as api, numpy as np, pandas as pd

saved = api.load_saved("e013_storeprod.parquet")
tt = api.train_targets()
df = saved.merge(tt, on=["household_key","snapshot_day"])
df = df[df.future_spend_4w.notna()]
num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number]).columns.tolist()
X = np.nan_to_num(df[num].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
y = df.future_spend_4w.values
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
Xs = (X-mu)/sd
def ridge(lam):
    A = Xs[tr]; I = np.eye(A.shape[1]); I[0,0]=0
    w = np.linalg.solve(A.T@A + lam*I, A.T@y[tr])
    return Xs@w
best=None
for lam in [1,10,100,300,1000,3000]:
    p = ridge(lam); mae = np.abs(p[va]-y[va]).mean()
    if best is None or mae<best[1]: best=(lam,mae)
print("proxy full E013: lam=%d val431 MAE %.3f" % best)
p = ridge(best[0]); res = y-p
print("resid mean %.2f std %.2f" % (res.mean(), res.std()))
for c in ["spend_4w_lag1","te2_ewm","te2_mean","te_hh_shrunk","spend_4w_recent","te2_slope"]:
    print("corr(resid,%s)=%.3f" % (c, np.corrcoef(res, df[c].fillna(0))[0,1]))

# single-feature baselines
for c in ["spend_4w_lag1","te2_ewm","te2_mean","te_hh_shrunk","spend_4w_recent"]:
    v = df[c].fillna(df[c].median()).values
    print("single %-16s val431 MAE %.2f" % (c, np.abs(v[va]-y[va]).mean()))
print("mean-only MAE %.2f" % np.abs(np.full(va.sum(), y[tr].mean())-y[va]).mean())
