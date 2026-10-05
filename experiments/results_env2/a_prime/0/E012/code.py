import agent_api as A, pandas as pd, numpy as np
te = A.load_saved("e007_te.parquet")
cols = [c for c in te.columns if c not in ("household_key","snapshot_day")]
print("E007 rows:", len(te), "feat cols:", len(cols))
for i in range(0, len(cols), 4):
    print(" | ".join(cols[i:i+4]))

tt = A.train_targets()
y = tt[A.TARGET]
print("\nTARGET describe:\n", y.describe().round(2))
print("zero share:", round((y==0).mean(),3))
print("\nper-snapshot target mean/median/count:")
print(tt.groupby("snapshot_day")[A.TARGET].agg(["mean","median","count"]).round(1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values

# Baselines: what do simple predictors achieve in MAE?
def mae(p): return np.mean(np.abs(y - p))
print("global median:", round(mae(np.median(y)),2))
print("global mean:", round(mae(y.mean()),2))
for c in ["te_hh_shrunk","te_hh_mean","spend_4w","spend_8w","spend_4w_lag1","te_prior"]:
    if c in df: print(f"{c:14s}", round(mae(df[c].values),2))

# blend of te and spend_4w (manual search) as upper-bound check
best=None
for w in np.linspace(0,1,21):
    p = w*df["te_hh_shrunk"].values + (1-w)*df["spend_4w"].values
    m = mae(p)
    if best is None or m<best[0]: best=(m,w)
print("best blend te/spend4w:", round(best[0],2), "w=",round(best[1],2))

# error decomposition of te_hh_shrunk: where is loss?
p = df["te_hh_shrunk"].values
e = np.abs(y-p)
df["err"] = e
print("\nMAE by target bucket:")
df["yb"] = pd.qcut(y, 10, duplicates="drop")
print(df.groupby("yb", observed=True).agg(n=("err","size"), mae=("err","mean"), pred=("te_hh_shrunk","mean"), true=(A.TARGET,"mean")).round(1))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
print("te rows:", len(te), "train rows:", len(tt), "merged:", len(df))
for c in ["te_hh_mean","te_hh_shrunk","te_bin","te_prior","te_home","te_kid","te_size","te_hh_n"]:
    print(c, "n_missing:", int(df[c].isna().sum()))
# check key match
print("keys equal:", set(map(tuple,te[["household_key","snapshot_day"]].values))==set(map(tuple,tt[["household_key","snapshot_day"]].values)))
# maybe te table has different snapshot_day dtype
print(te.dtypes.head(6))
print(te.head(3)[["household_key","snapshot_day","te_hh_mean","te_hh_shrunk"]])
print(tt.head(3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")

# per-snapshot missing te
print(df.groupby("snapshot_day")["te_hh_shrunk"].apply(lambda s: s.isna().sum()))

# single-feature baselines with fallback chain
y = df[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))
med = np.median(y)
p_te = df["te_hh_shrunk"].fillna(df["spend_4w_lag1"]).fillna(med).values
print("te chain MAE:", round(mae(p_te),2))
p_lag1 = df["spend_4w_lag1"].fillna(med).values
print("lag1 MAE:", round(mae(p_lag1),2))
for w in np.linspace(0,1,11):
    print(f"w={w:.1f}", round(mae(w*p_te+(1-w)*p_lag1),2))

# zero-spend rows: what predicts them?
z = y==0
print("\nzero rows:", z.sum(), "share", round(z.mean(),3))
for c in ["days_since_last","active_4w","trend_4_28","spend_4w","nbask_4w","spend_4w_lag1","gap_mean_112","tenure_days"]:
    print(f"{c:18s} zero-mean={df.loc[z,c].mean():8.2f}  nonzero-mean={df.loc[~z,c].mean():8.2f}")


# ---- cell ----
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


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values

# correlation of target with candidate predictors (on train rows)
cands = ["te_hh_shrunk","te_hh_mean","spend_4w","spend_8w","spend_12w","spend_28w","spend_56w","spend_112w",
         "spend_4w_lag1","spend_4w_lag2","spend_4w_lag3","spend_112_mean4","spend_112_std","spend_112_cv",
         "nbask_4w","nbask_8w","nbask_112","days_since_last","tenure_days","avg_basket_12w",
         "spend_per_basket_112","trend_4_8","trend_4_28","gap_mean_112","gap_max_112","gap_std_112",
         "n_campaign_targets","days_since_last_tgt_start","share_disp_28","share_mail_28","te_prior","te_bin"]
rows=[]
for c in cands:
    if c in df:
        x = df[c].astype(float)
        m = x.notna() & ~np.isnan(y)
        if m.sum()>100:
            rows.append((c, round(float(np.corrcoef(x[m], y[m])[0,1]),3)))
rows.sort(key=lambda r:-abs(r[1]))
for r in rows: print(r)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values

# Ridge on standardized numeric features, fit on train snapshots <=403, validate on 431
feat = [c for c in te.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
Xdf = df[num].astype(float)
mu, sd = Xdf.mean(), Xdf.std().replace(0,1)
X = ((Xdf-mu)/sd).fillna(0).values
yv = y.copy()

tr = df["snapshot_day"] <= 403
va = df["snapshot_day"] == 431
Xtr, ytr, Xva, yva = X[tr], yv[tr], X[va], yv[va]
def ridge(Xtr, ytr, Xva, lam):
    A_ = Xtr.T@Xtr + lam*np.eye(Xtr.shape[1])
    b = np.linalg.solve(A_, Xtr.T@ytr)
    return Xva@b
print("ridge val MAE at 431 (lam):")
for lam in [300, 1000, 3000, 10000, 30000, 100000]:
    p = ridge(Xtr, ytr, Xva, lam)
    print(f"  lam={lam:>6}", round(float(np.mean(np.abs(yva-p))),2))
# global median baseline at 431
print("median pred MAE at 431:", round(float(np.mean(np.abs(yva-np.median(ytr)))),2))
# lag1-only
lag1 = df.loc[va,"spend_4w_lag1"].fillna(df.loc[tr,"spend_4w_lag1"].median()).values
print("lag1 MAE at 431:", round(float(np.mean(np.abs(yva-lag1))),2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values

feat = [c for c in te.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
Xdf = df[num].astype(float)
mu, sd = Xdf.mean(), Xdf.std().replace(0,1)
X = ((Xdf-mu)/sd).fillna(0).values

tr = df["snapshot_day"] <= 403
va = df["snapshot_day"] == 431
ytr, yva = y[tr], y[va]

def ridge_solve(Xtr, Ytr, lam):
    A_ = Xtr.T@Xtr + lam*np.eye(Xtr.shape[1])
    return np.linalg.solve(A_, Xtr.T@Ytr)

# log-target ridge
ly = np.log1p(y)
for lam in [100, 300, 1000, 3000, 10000]:
    b = ridge_solve(X[tr], ly[tr], lam)
    p = np.expm1(np.clip(X[va]@b, 0, 8))
    print(f"log-target ridge lam={lam:>5} MAE@431:", round(float(np.mean(np.abs(yva-p))),2))

# two-part: logistic zero + ridge on log for nonzero
z_tr = (ytr==0).astype(float)
Xz = np.c_[np.ones(tr.sum()), X[tr]]
beta = np.zeros(Xz.shape[1])
for _ in range(500):
    p = 1/(1+np.exp(-Xz@beta))
    beta -= 0.05/Xz.shape[0]*(Xz.T@(p-z_tr))
Xvz = np.c_[np.ones(va.sum()), X[va]]
pz = 1/(1+np.exp(-Xvz@beta))
b = ridge_solve(X[tr][ytr>0], ly[tr][ytr>0], 1000)
pamt = np.expm1(np.clip(Xvz@b, 0, 8))
p2 = pz*pamt
print("two-part MAE@431:", round(float(np.mean(np.abs(yva-p2))),2))
# blend with lag1
lag1 = df.loc[va,"spend_4w_lag1"].fillna(df.loc[tr,"spend_4w_lag1"].median()).values
for w in [0,0.25,0.5,0.75,1.0]:
    print(f"blend two-part/lag1 w={w}:", round(float(np.mean(np.abs(yva-(w*p2+(1-w)*lag1)))),2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
feat = [c for c in te.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
Xdf = df[num].astype(float)
mu, sd = Xdf.mean(), Xdf.std().replace(0,1)
X = ((Xdf-mu)/sd).fillna(0).values
tr = df["snapshot_day"] <= 403
va = df["snapshot_day"] == 431
ytr, yva = y[tr], y[va]
ly = np.log1p(y)
Xz = np.c_[np.ones(tr.sum()), X[tr]]
z_tr = (ytr==0).astype(float)
beta = np.zeros(Xz.shape[1])
for _ in range(500):
    p = 1/(1+np.exp(-Xz@beta))
    beta -= 0.05/Xz.shape[0]*(Xz.T@(p-z_tr))
Xvz = np.c_[np.ones(va.sum()), X[va]]
pz = 1/(1+np.exp(-Xvz@beta))
mask = ytr>0
b = np.linalg.solve(X[tr][mask].T@X[tr][mask] + 1000*np.eye(X.shape[1]), X[tr][mask].T@ly[tr][mask])
pamt = np.expm1(np.clip(Xvz@b, 0, 8))
p2 = pz*pamt
print("two-part MAE@431:", round(float(np.mean(np.abs(yva-p2))),2))
lag1 = df.loc[va,"spend_4w_lag1"].fillna(df.loc[tr,"spend_4w_lag1"].median()).values
for w in [0,0.25,0.5,0.75,1.0]:
    print(f"blend two-part/lag1 w={w}:", round(float(np.mean(np.abs(yva-(w*p2+(1-w)*lag1)))),2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
feat = [c for c in te.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
print("n num cols:", len(num), "rows:", len(df))
Xdf = df[num].astype(float)
print("Xdf shape:", Xdf.shape)
mu, sd = Xdf.mean(), Xdf.std().replace(0,1)
X = ((Xdf-mu)/sd).fillna(0).values
print("X shape:", X.shape)
tr = (df["snapshot_day"] <= 403).values
va = (df["snapshot_day"] == 431).values
print("tr sum:", tr.sum(), "va sum:", va.sum(), "X[tr] shape:", X[tr].shape)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
feat = [c for c in te.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(df[c])]
Xdf = df[num].astype(float)
mu, sd = Xdf.mean(), Xdf.std().replace(0,1)
X = ((Xdf-mu)/sd).fillna(0).values
tr = (df["snapshot_day"] <= 403).values
va = (df["snapshot_day"] == 431).values
ytr, yva = y[tr], y[va]
ly = np.log1p(y)
Xz = np.c_[np.ones(tr.sum()), X[tr]]
z_tr = (ytr==0).astype(float)
beta = np.zeros(Xz.shape[1])
for _ in range(500):
    p = 1/(1+np.exp(-Xz@beta))
    beta -= 0.05/Xz.shape[0]*(Xz.T@(p-z_tr))
Xvz = np.c_[np.ones(va.sum()), X[va]]
pz = 1/(1+np.exp(-Xvz@beta))
mask = ytr>0
b = np.linalg.solve(X[tr][mask].T@X[tr][mask] + 1000*np.eye(X.shape[1]), X[tr][mask].T@ly[tr][mask])
pamt = np.expm1(np.clip(Xvz[:,1:]@b, 0, 8))
p2 = pz*pamt
print("two-part MAE@431:", round(float(np.mean(np.abs(yva-p2))),2))
lag1 = df.loc[va,"spend_4w_lag1"].fillna(df.loc[tr,"spend_4w_lag1"].median()).values
for w in [0,0.25,0.5,0.75,1.0]:
    print(f"blend two-part/lag1 w={w}:", round(float(np.mean(np.abs(yva-(w*p2+(1-w)*lag1)))),2))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))

# ---- leak-free household outcome history (median / p_zero / ewm) ----
v = A.snapshot(459)  # capped view; enough to compute outcomes for windows ending <= 459
tr_days = A.snapshot_days()["train"]
tx = v.table("transactions")
sp = tx.groupby(["household_key","day"])["sales_value"].sum().reset_index()
# outcome for snapshot s' = spend in (s', s'+28]
def outcome_at(sp_end):
    m = (sp["day"] > sp_end-28) & (sp["day"] <= sp_end)
    g = sp[m].groupby("household_key")["sales_value"].sum()
    return g
outs = {}  # snapshot_day -> series of outcomes (only for train snapshot days whose window ends <= 459)
for s in tr_days:
    if s+28 <= 459:
        outs[s] = outcome_at(s+28)
days_sorted = sorted(outs)
# per household build history lists
hh_hist = {}
for s in days_sorted:
    o = outs[s]
    for h, val in o.items():
        hh_hist.setdefault(h, []).append((s, float(val)))

rows = []
for h, s in zip(df["household_key"], df["snapshot_day"]):
    hist = hh_hist.get(h, [])
    past = [val for (d, val) in hist if d < s]
    if past:
        med = float(np.median(past)); mean = float(np.mean(past))
        pz = float(np.mean([x==0 for x in past]))
        w = np.array([0.7**(len(past)-1-i) for i in range(len(past))])
        ewm = float(np.average(past, weights=w))
    else:
        med = mean = ewm = np.nan; pz = np.nan
    rows.append((med, mean, pz, ewm))
arr = np.array(rows, dtype=float)
df["te_med"] = arr[:,0]; df["te_mean_chk"] = arr[:,1]; df["p_zero"] = arr[:,2]; df["te_ewm"] = arr[:,3]
print("corr te_mean_chk vs te_hh_mean:", round(float(np.corrcoef(df["te_mean_chk"].fillna(-1), df["te_hh_mean"].fillna(-1))[0,1]),4))

med_g = float(np.median(y))
print("\nsingle-predictor MAE (train rows):")
print("te_hh_mean (E007):", round(mae(df["te_hh_mean"].fillna(df["spend_4w_lag1"]).fillna(med_g)),2))
print("te_med  (median):", round(mae(df["te_med"].fillna(df["spend_4w_lag1"]).fillna(med_g)),2))
print("te_ewm:", round(mae(df["te_ewm"].fillna(df["spend_4w_lag1"]).fillna(med_g)),2))
# two-part style: (1-p_zero)*median_pos
pos_med = df["te_med"].where(df["p_zero"]<1)  # median over positive outcomes
# compute median of positive past outcomes
rows2=[]
for h, s in zip(df["household_key"], df["snapshot_day"]):
    past=[val for (d,val) in hh_hist.get(h,[]) if d<s and val>0]
    rows2.append(float(np.median(past)) if past else np.nan)
df["te_medpos"]=np.array(rows2)
p2 = (1-df["p_zero"].fillna(0.207))*df["te_medpos"].fillna(med_g)
print("two-part (1-pz)*medpos:", round(mae(p2.values),2))
for w in [0.25,0.5,0.75]:
    print(f"blend two-part/te_med w={w}:", round(mae(w*p2.values+(1-w)*df["te_med"].fillna(med_g).values),2))

# ---- cohort drift ----
coh = df.groupby("snapshot_day").agg(trail=("spend_4w","mean"), ymean=(A.TARGET,"mean"), ymed=(A.TARGET,"median"))
print("\ncohort per snapshot:\n", coh.round(1))
print("corr trailing cohort spend_4w vs next-window target mean:", round(float(np.corrcoef(coh["trail"][:-1], coh["ymean"][1:])[0,1]),3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))
med_g = float(np.median(y))

# E011's outcome-history features already exist; load and compare
e11 = A.load_saved("e011_outcomehist.parquet")
e11cols = [c for c in e11.columns if c not in set(te.columns)|{"household_key","snapshot_day"}]
print("E011 added cols:", e11cols)
d11 = df.merge(e11[["household_key","snapshot_day"]+e11cols], on=["household_key","snapshot_day"], how="left")
for c in e11cols:
    x = d11[c].astype(float)
    m = x.notna()
    print(f"{c:22s} corr={np.corrcoef(x[m], y[m])[0,1]:.3f}  mae_single={mae(x.fillna(df['spend_4w_lag1']).fillna(med_g)):.2f}")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))
med_g = float(np.median(y))
lag1 = df["spend_4w_lag1"].fillna(med_g).values
te_hh = df["te_hh_mean"].fillna(df["spend_4w_lag1"]).fillna(med_g).values

# how much does te_hh_mean improve over lag1? and does blending help?
print("lag1:", round(mae(lag1),2), " te_hh_mean:", round(mae(te_hh),2))
for w in [0,0.25,0.5,0.75,1.0]:
    print(f"blend te/lag1 w={w}:", round(mae(w*te_hh+(1-w)*lag1),2))

# E011's oh_* features as single predictors (numeric only)
e11 = A.load_saved("e011_outcomehist.parquet")
e11cols = [c for c in e11.columns if c.startswith("oh_")]
d11 = df.merge(e11[["household_key","snapshot_day"]+e11cols], on=["household_key","snapshot_day"], how="left")
for c in e11cols:
    x = d11[c].astype(float)
    m = x.notna()
    if m.sum()>100:
        print(f"{c:22s} corr={np.corrcoef(x[m], y[m])[0,1]:.3f}  mae_single={mae(x.fillna(df['spend_4w_lag1']).fillna(med_g)):.2f}")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

TRAIN_DAYS = list(range(95, 432, 28))
ANCHOR_DAYS = TRAIN_DAYS + [459, 487]

def build(view, s):
    tx = view.table("transactions")
    hs = np.asarray(view.households).ravel()
    # ---- household-day spend table (sorted by hh, day) ----
    g = tx.groupby(["household_key","day"], sort=True)["sales_value"].sum().reset_index()
    hh_arr = g["household_key"].values
    day_arr = g["day"].values.astype(np.int64)
    sp_arr = g["sales_value"].values.astype(float)
    uniq = np.unique(hh_arr)
    starts = np.searchsorted(hh_arr, uniq, side="left")
    ends = np.searchsorted(hh_arr, uniq, side="right")
    csg = np.concatenate([[0.0], np.cumsum(sp_arr)])
    h2i = {int(h): i for i, h in enumerate(uniq)}
    # ---- basket-level table (sorted by hh, day) ----
    bk = tx.groupby("basket_id", as_index=False).agg(hh=("household_key","first"),
                                                     day=("day","first"),
                                                     sp=("sales_value","sum"))
    bk = bk.sort_values(["hh","day"], kind="stable")
    bhh = bk["hh"].values; bday = bk["day"].values.astype(np.int64); bsp = bk["sp"].values.astype(float)
    buniq = np.unique(bhh)
    bstarts = np.searchsorted(bhh, buniq, side="left")
    bends = np.searchsorted(bhh, buniq, side="right")
    b2i = {int(h): i for i, h in enumerate(buniq)}
    anchors = [a for a in ANCHOR_DAYS if a + 28 <= s]
    rows = np.zeros((len(hs), 22), dtype=float)
    NEW = ["te2_n","te2_mean","te2_med","te2_std","te2_cv","te2_zero","te2_min","te2_max",
           "te2_slope","te2_ewm","gap_mean_all","gap_med_all","gap_std_all","gap_mean_8w",
           "exp_trips_4w","exp_spend_4w","exp_spend_112","spend_7d","nbask_7d","active_7d",
           "max_basket_4w","nbask_ratio_4_28"]
    for r, h in enumerate(hs):
        i = h2i.get(int(h)); bi = b2i.get(int(h))
        if i is None:
            continue
        a0, a1 = starts[i], ends[i]
        d = day_arr[a0:a1]; fd = d[0]
        # outcome series over eligible anchors
        vals = []
        for anc in anchors:
            if fd <= anc - 84:
                j = np.searchsorted(d, anc, side="right")
                k = np.searchsorted(d, anc+28, side="right")
                vals.append(csg[a0+k] - csg[a0+j])
        n = len(vals)
        if n:
            v = np.array(vals); m = v.mean()
            rows[r,0] = n; rows[r,1] = m; rows[r,2] = np.median(v); rows[r,3] = v.std()
            rows[r,4] = v.std()/m if m > 0 else np.nan
            rows[r,5] = (v==0).mean(); rows[r,6] = v.min(); rows[r,7] = v.max()
            k3 = min(3, n)
            rows[r,8] = v[-k3:].mean() - v[:k3].mean()
            w = 0.7 ** np.arange(n-1, -1, -1)
            rows[r,9] = (v*w).sum()/w.sum()
        if len(d) > 1:
            gp = np.diff(d).astype(float)
            rows[r,10] = gp.mean(); rows[r,11] = np.median(gp); rows[r,12] = gp.std()
        d8 = d[d > s-56]
        if len(d8) > 1: rows[r,13] = np.diff(d8).mean()
        d112 = d[d > s-112]
        gm112 = np.diff(d112).mean() if len(d112) > 1 else np.nan
        # expected trips x basket size
        gm = rows[r,10]
        if bi is not None:
            b0, b1 = bstarts[bi], bends[bi]
            bd = bday[b0:b1]; bs = bsp[b0:b1]
            if np.isfinite(gm) and gm > 0:
                rows[r,14] = 28.0/gm
                rows[r,15] = 28.0/gm * bs.mean()
            if np.isfinite(gm112) and gm112 > 0:
                j = np.searchsorted(bd, s-784, side="right")
                nb112 = b1-b0-j
                if nb112 > 0:
                    sp112 = csg[a0+np.searchsorted(d, s-784, side="right")] - csg[a0] if len(d) else 0.0
                    rows[r,16] = 28.0/gm112 * (sp112/nb112)
            j7 = np.searchsorted(bd, s-7, side="right"); n7 = b1-b0-j7
            rows[r,18] = n7; rows[r,19] = 1.0 if n7 > 0 else 0.0
            j4 = np.searchsorted(bd, s-28, side="right"); n4 = b1-b0-j4
            rows[r,20] = bs[j4:].max() if n4 > 0 else 0.0
            j28 = np.searchsorted(bd, s-196, side="right"); n28 = b1-b0-j28
            if n28 > 0: rows[r,21] = n4/(n28/7.0)
        rows[r,17] = csg[a0+np.searchsorted(d, s-7, side="right")] - csg[a0]
    out = pd.DataFrame(rows, index=pd.Index(hs, name="household_key"), columns=NEW)
    if s == TRAIN_DAYS[0]:
        print("snap", s, "hh", len(hs), "anchors", len(anchors), "te2_n>0 share", round(float((out["te2_n"]>0).mean()),3))
    return out

feats = A.build_features(build)
print("feats:", feats.shape)
e7 = A.load_saved("e007_te.parquet")
full = e7.merge(feats, on=["household_key","snapshot_day"], how="inner")
print("full:", full.shape)
assert len(full) == 36426
tt = A.train_targets()
dg = tt.merge(full, on=["household_key","snapshot_day"], how="left")
y = dg[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))
med = float(np.median(y))
lag1 = dg["spend_4w_lag1"].fillna(med).values
print("corr te2_mean~y:", round(float(np.corrcoef(dg["te2_mean"].fillna(0), y)[0,1]),3),
      " corr te2_mean~te_hh_mean:", round(float(np.corrcoef(dg["te2_mean"].fillna(0), dg["te_hh_mean"].fillna(0))[0,1]),3))
print("single MAE te2_mean chain:", round(mae(dg["te2_mean"].fillna(dg["spend_4w_lag1"]).fillna(med).values),2),
      " te_hh_mean chain:", round(mae(dg["te_hh_mean"].fillna(dg["spend_4w_lag1"]).fillna(med).values),2))
print("te2_zero NaNs:", int(dg["te2_zero"].isna().sum()), " exp_spend_4w corr:",
      round(float(np.corrcoef(dg["exp_spend_4w"].fillna(0), y)[0,1]),3))
path = A.save_table(full, "e012_outcome2.parquet")
print("saved:", path)
