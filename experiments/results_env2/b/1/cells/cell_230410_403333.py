
import pandas as pd, numpy as np, agent_api

base = load_saved("e008_level_shape.parquet")
t = train_targets()
df = base.merge(t, on=["household_key","snapshot_day"], how="inner")
y = df[TARGET].values.astype(float)
feat_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
X = df[feat_cols].copy()
cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
num_cols = [c for c in feat_cols if c not in cat_cols]
Xn = X[num_cols].apply(pd.to_numeric, errors="coerce")
for i, c in enumerate(cat_cols):
    Xn = pd.concat([Xn, pd.get_dummies(X[c].astype("category"), prefix=f"cat{i}", dummy_na=True).astype(float)], axis=1)
Xn = Xn.fillna(Xn.median())
Xv = Xn.values.astype(np.float64); n = len(y)

# leak-free candidate features from per-household day-sorted tx (filter day <= s per row)
snap = agent_api.snapshot(459)
tx = snap.transactions[["household_key","day","sales_value"]].sort_values(["household_key","day"])
keys = tx.household_key.values; days = tx.day.values.astype(int); sv = tx.sales_value.values.astype(float)
uniq, start = np.unique(keys, return_index=True)
h2idx = {h:i for i,h in enumerate(uniq)}
ends = np.r_[start[1:], len(keys)]
cs = np.concatenate([[0.0], np.cumsum(sv)])
def win_sum(h, lo, hi):
    i = h2idx.get(h)
    if i is None: return 0.0
    a = np.searchsorted(days[start[i]:ends[i]], lo, "left") + start[i]
    b = np.searchsorted(days[start[i]:ends[i]], hi, "right") + start[i]
    return cs[b]-cs[a]
def win_cnt(h, lo, hi):
    i = h2idx.get(h)
    if i is None: return 0.0
    a = np.searchsorted(days[start[i]:ends[i]], lo, "left")
    b = np.searchsorted(days[start[i]:ends[i]], hi, "right")
    return float(b-a)
def act_weeks(h, lo, hi):
    i = h2idx.get(h)
    if i is None: return 0.0
    dd = days[start[i]:ends[i]]
    m = (dd > lo) & (dd <= hi)
    return len(np.unique((dd[m]+8)//7)) if m.any() else 0.0

hh = df.household_key.values; sd = df.snapshot_day.values.astype(int)
sp28v = df.sp28.values.astype(float); sp56v = df.sp56.values.astype(float)
sp84v = df.sp84.values.astype(float); sp364v = df.sp364.values.astype(float)
seas1y = np.array([win_sum(h, s-364, s-336) for h, s in zip(hh, sd)])
aw4  = np.array([act_weeks(h, s-28, s) for h, s in zip(hh, sd)])
aw52 = np.array([act_weeks(h, s-364, s) for h, s in zip(hh, sd)])
ten  = df.tenure.values.astype(float)

eps = 1.0
cand = {}
cand["rr28_364"] = sp28v / np.maximum(sp364v/13.0, eps)
cand["rr28_84"]  = sp28v / np.maximum(sp84v/3.0, eps)
cand["rr56_364"] = sp56v / np.maximum(sp364v/6.5, eps)
cand["rr28_1y"]  = sp28v / np.maximum(seas1y, eps)
cand["aw_ratio"] = aw4 / np.maximum(aw52/13.0, 0.5)
cand["ten_rate"] = sp364v / np.maximum(ten, 28.0)
cand["med4w_x_rr"] = df.z_med4w_hist.values * np.clip(cand["rr28_364"], 0, 3)
for k in list(cand):
    cand["log_"+k] = np.log1p(np.clip(cand[k], 0, 50))

idx = np.random.RandomState(0).permutation(n); K=5
def ridge_fit(Xtr,ytr,a=100):
    mu=Xtr.mean(0); sdv=Xtr.std(0)+1e-9
    A=np.hstack([(Xtr-mu)/sdv,np.ones((len(Xtr),1))])
    G=A.T@A; p=G.shape[0]
    Gg=G.copy(); Gg[np.arange(p),np.arange(p)]+=a*np.r_[np.ones(p-1),0]
    return np.linalg.solve(Gg,A.T@ytr),mu,sdv
def cv_mae(Xall):
    s=0.0
    for k in range(K):
        va=idx[k::K]; tr=np.setdiff1d(idx,va)
        w,mu,sdv=ridge_fit(Xall[tr],y[tr])
        p=np.hstack([(Xall[va]-mu)/sdv,np.ones((len(va),1))])@w
        s+=np.abs(y[va]-p).sum()
    return round(s/n,2)
print("base ridge:", cv_mae(Xv))
# add candidates one block at a time
C = np.column_stack([cand[k] for k in cand])
print("with all candidates:", cv_mae(np.hstack([Xv, C])))
for k in cand:
    print(f"  +{k:14s}: {cv_mae(np.hstack([Xv, cand[k][:,None]]))}")
# correlation of candidates with y
print("\ncand corr with y:", {k: round(float(np.corrcoef(v,y)[0,1]),3) for k,v in cand.items()})
