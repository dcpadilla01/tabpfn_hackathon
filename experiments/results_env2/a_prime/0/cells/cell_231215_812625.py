import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values

# zero-class separability: AUC of single features for y==0
from numpy import argsort
def auc(feat, z):
    f = df[feat].fillna(df[feat].median()).values
    order = np.argsort(f)
    r = np.empty(len(f)); r[order] = np.arange(1,len(f)+1)
    n1, n0 = z.sum(), (~z).sum()
    return (r[z].sum() - n1*(n1+1)/2) / (n1*n0)
print("AUC for predicting zero-spend (higher feat -> more likely zero):")
for c in ["days_since_last","active_4w","spend_4w","nbask_4w","spend_4w_lag1","gap_mean_112",
          "gap_max_112","spend_112_cv","trend_4_28","nbask_8w","active_4w","spend_112_std","te_hh_n"]:
    if c in df: print(f"  {c:18s} {auc(c, y==0):.3f}")

# how well can we predict the zero class with a simple logistic-like rule? quick check:
f = df[["days_since_last","active_4w","spend_4w","gap_mean_112","spend_4w_lag1"]].copy()
for c in f: f[c]=f[c].fillna(f[c].median())
z = (y==0).astype(float)
# crude logistic regression via numpy (few iters)
X = np.c_[np.ones(len(f)), f.values]
beta = np.zeros(X.shape[1])
for _ in range(300):
    p = 1/(1+np.exp(-X@beta))
    beta -= 0.01/X.shape[0]*(X.T@(p-z))
print("zero-class model AUC (in-sample):", round(float(np.corrcoef(X@beta, z)[0,1]),3), "mean p:", round(float(p.mean()),3))
