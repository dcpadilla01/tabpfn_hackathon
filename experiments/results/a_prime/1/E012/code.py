import agent_api as A

names = ["e011_demo.parquet","e012_dorm.parquet","e013_peers.parquet",
         "micro.parquet","nf_candidates.parquet","nf_compact19.parquet",
         "nf_p1.parquet","nf_robust.parquet","nf_seasonal.parquet","nf_transforms.parquet"]
for n in names:
    try:
        df = A.load_saved(n)
        print("==", n, df.shape)
        print(list(df.columns))
    except Exception as e:
        print("==", n, "ERR", repr(e))


# ---- cell ----
import agent_api as A
import pandas as pd

for n in ["e013_peers.parquet","micro.parquet","nf_candidates.parquet"]:
    df = A.load_saved(n)
    print("==", n, df.shape)
    for c in df.columns:
        print("  ", c, df[c].dtype)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

tt = A.train_targets()
base_cols = set(A.load_saved("e011_demo.parquet").columns)

groups = {
 "dorm": [c for c in A.load_saved("e012_dorm.parquet").columns if c not in base_cols],
 "peers": [c for c in A.load_saved("e013_peers.parquet").columns if c not in base_cols],
 "micro": [c for c in A.load_saved("micro.parquet").columns if c not in base_cols],
 "robust": [c for c in A.load_saved("nf_robust.parquet").columns if c not in base_cols],
 "seasonal": [c for c in A.load_saved("nf_seasonal.parquet").columns if c not in base_cols],
}
print({k: len(v) for k,v in groups.items()})

tt2 = tt.merge(A.load_saved("e011_demo.parquet")[["household_key","snapshot_day","spend_rate28","spend_l1","zero_recent"]],
               on=["household_key","snapshot_day"])
y = tt2["future_spend_4w"]

for g, cols in groups.items():
    df = A.load_saved({  "dorm":"e012_dorm.parquet","peers":"e013_peers.parquet","micro":"micro.parquet",
                         "robust":"nf_robust.parquet","seasonal":"nf_seasonal.parquet"}[g])
    m = tt2.merge(df[["household_key","snapshot_day"]+cols], on=["household_key","snapshot_day"], how="left")
    rows=[]
    for c in cols:
        x = m[c].astype(float)
        r1 = x.corr(y)
        # partial: corr of residual after regressing on spend_rate28
        b = np.polyfit(m["spend_rate28"].fillna(0), y, 1)
        res = y - np.polyval(b, m["spend_rate28"].fillna(0))
        r2 = x.corr(res)
        rows.append((c, round(r1,3), round(r2,3)))
    rows.sort(key=lambda t: -abs(t[2]))
    print("==", g)
    for r in rows[:12]: print("  ", r)


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

e11 = A.load_saved("e011_demo.parquet")
tt = A.train_targets()
m = tt.merge(e11[["household_key","snapshot_day","spend_l1","spend_l2","spend_rate28","zero_recent","tenure"]],
             on=["household_key","snapshot_day"])

micro = A.load_saved("micro.parquet")[["household_key","snapshot_day","mspend_l1","spend_7d","spend_14d","trips_14d"]]
m = m.merge(micro, on=["household_key","snapshot_day"], how="left")
print("corr mspend_l1 vs spend_l1:", m["mspend_l1"].corr(m["spend_l1"]))
print(m[["spend_l1","mspend_l1","spend_7d","spend_14d","trips_14d"]].describe().round(2))

peers = A.load_saved("e013_peers.parquet")[["household_key","snapshot_day","peer_ratio","peer_recent28","spend_ly4w","nratio_ly" if "nratio_ly" in A.load_saved("e013_peers.parquet").columns else "peer_p90"]]
m = m.merge(peers, on=["household_key","snapshot_day"], how="left")
print("\npeer_ratio describe:"); print(m["peer_ratio"].describe().round(3))
print("corr peer_recent28 vs spend_l1:", m["peer_recent28"].corr(m["spend_l1"]))
print(m[["peer_ratio","peer_recent28","spend_ly4w"]].head(8).round(3))

seas = A.load_saved("nf_seasonal.parquet")
m = m.merge(seas, on=["household_key","snapshot_day"], how="left")
print("\nnratio_ly describe:"); print(m["nratio_ly"].describe().round(3))
print(m[["spend_l1","nratio_ly","nspend_ly","nseas_uplift"]].head(8).round(3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

tt = A.train_targets()
base = A.load_saved("e011_demo.parquet")
dorm = A.load_saved("e012_dorm.parquet")
micro = A.load_saved("micro.parquet")
rob = A.load_saved("nf_robust.parquet")

base_cols = [c for c in base.columns if c not in ("household_key","snapshot_day")]
cat_cols = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]
num_base = [c for c in base_cols if c not in cat_cols]

dorm_cols = ["inactive_run84","gap_last","gap_mean84","gap_std84","gap_max84","n_zero_l6","active_week_share","zero_frac_full","tenure_x","tenure_y"]
micro_new = ["trips_7d","trips_14d","n_zero_w6","n_zero_w12","spend_cv_l6","spend_max_over_med_l6","gap_mean_l6","gap_std_l6","gap_max_l6","n_gap21_l6",
             "basket_med_84","basket_max_84","basket_cv_84","basket_max_over_med_84","n_stockup_84","dsl_stockup","stockup_share_84","units_84",
             "unit_price_84","units_per_trip_84","weekend_share_84","stores_84","morning_share_84"]
rob_cols = [c for c in rob.columns if c not in ("household_key","snapshot_day")]

m = tt.merge(base[["household_key","snapshot_day"]+num_base], on=["household_key","snapshot_day"], how="left")
# one-hot demographics
d = base[["household_key","snapshot_day"]+cat_cols]
oh = pd.get_dummies(d[cat_cols].astype("category"), dummy_na=True, prefix=cat_cols)
oh["household_key"]=d["household_key"].values; oh["snapshot_day"]=d["snapshot_day"].values
m = m.merge(oh, on=["household_key","snapshot_day"], how="left")
m = m.merge(dorm[["household_key","snapshot_day"]+dorm_cols], on=["household_key","snapshot_day"], how="left")
m = m.merge(micro[["household_key","snapshot_day"]+micro_new], on=["household_key","snapshot_day"], how="left")
m = m.merge(rob[["household_key","snapshot_day"]+rob_cols], on=["household_key","snapshot_day"], how="left")

y = m["future_spend_4w"].values.astype(float)
tr_days = A.snapshot_days()["train"]; va_days = A.snapshot_days()["validation"]
is_tr = m["snapshot_day"].isin(tr_days).values; is_va = m["snapshot_day"].isin(va_days).values

def make_X(cols):
    X = m[cols].astype(float).copy()
    return X.values

groups = {"base": num_base+list(oh.columns),
          "dorm": num_base+list(oh.columns)+dorm_cols,
          "micro": num_base+list(oh.columns)+micro_new,
          "robust": num_base+list(oh.columns)+rob_cols,
          "dorm+micro": num_base+list(oh.columns)+dorm_cols+micro_new,
          "dorm+robust": num_base+list(oh.columns)+dorm_cols+rob_cols,
          "micro+robust": num_base+list(oh.columns)+micro_new+rob_cols,
          "all": num_base+list(oh.columns)+dorm_cols+micro_new+rob_cols}

def ridge_eval(cols, alphas=(0.1,1,10,100,1000)):
    X = make_X(cols)
    mu = X[is_tr].mean(0); sd = X[is_tr].std(0)+1e-9
    Xs = (X-mu)/sd
    Xtr, ytr = Xs[is_tr], y[is_tr]
    Xv = Xs[is_va]
    # LOSO alpha pick on train snapshots
    days_tr = m.loc[is_tr,"snapshot_day"].values
    uniq = np.unique(days_tr)
    best_a, best = None, 1e18
    for a in alphas:
        errs=[]
        for d0 in uniq:
            mtr = days_tr!=d0
            A_ = Xtr[mtr]; b_ = ytr[mtr]
            G = A_.T@A_ + a*np.eye(A_.shape[1]); G[0,0]-=a  # intercept unpenalized (first col? no intercept col) 
            # add intercept column
        # redo with intercept
        pass
    return None

# simpler: add intercept column, ridge with intercept unpenalized
def fit_ridge(Xtr, ytr, a):
    n,p = Xtr.shape
    Xd = np.hstack([np.ones((n,1)), Xtr])
    G = Xd.T@Xd + a*np.eye(p+1); G[0,0]-=a
    w = np.linalg.solve(G, Xd.T@ytr)
    return w

def pred(w, X):
    return w[0] + X@w[1:]

results={}
for name, cols in groups.items():
    X = make_X(cols)
    mu = np.nan_to_num(np.nanmean(X[is_tr],0)); 
    X = np.where(np.isnan(X), mu, X)
    sd = X[is_tr].std(0)+1e-9
    Xs = (X-mu)/sd
    Xtr, ytr = Xs[is_tr], y[is_tr]
    days_tr = m.loc[is_tr,"snapshot_day"].values
    uniq = np.unique(days_tr)
    best_a, best_score = None, 1e18
    for a in (1,10,100,1000):
        errs=[]
        for d0 in uniq:
            w = fit_ridge(Xtr[days_tr!=d0], ytr[days_tr!=d0], a)
            errs.append(np.mean(np.abs(pred(w, Xtr[days_tr==d0])-ytr[days_tr==d0])))
        s = np.mean(errs)
        if s<best_score: best_score, best_a = s, a
    w = fit_ridge(Xtr, ytr, best_a)
    pv = pred(w, Xs[is_va])
    mae = np.mean(np.abs(pv-y[is_va]))
    results[name]=(round(best_score,3), best_a, round(mae,3), len(cols))
    print(name, results[name])


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

e10 = A.load_saved("e010_l13fix.parquet")
e11 = A.load_saved("e011_demo.parquet")
e03 = A.load_saved("e003_catmix.parquet")
print("e010 cols not in e003:", [c for c in e10.columns if c not in e03.columns])
print("e011 cols not in e003:", [c for c in e11.columns if c not in e03.columns])
print("e010 shape", e10.shape, "e011 shape", e11.shape)
# check l13 NaN pattern in e010 vs e011
m = e11[["household_key","snapshot_day","spend_l13","tenure"]].merge(
    e10[["household_key","snapshot_day","spend_l13","has_real_l13","l13_over_recent"]],
    on=["household_key","snapshot_day"], suffixes=("_e11","_e10"))
print("e11 l13 NaN frac:", m["spend_l13_e11"].isna().mean().round(3), " e10:", m["spend_l13_e10"].isna().mean().round(3))
print(m[["snapshot_day","tenure","has_real_l13"]].drop_duplicates("snapshot_day").groupby("snapshot_day")["has_real_l13"].mean().round(2))

# target distribution / zero mass
tt = A.train_targets()
print("\ntarget describe:"); print(tt["future_spend_4w"].describe().round(2))
print("zero frac:", (tt["future_spend_4w"]==0).mean().round(3))


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

# Build E014: E010 + E011's demographic columns (independent mutations of E003 combined)
e10 = A.load_saved("e010_l13fix.parquet")
e11 = A.load_saved("e011_demo.parquet")
demo_cols = ["classification_1","classification_2","classification_3","classification_4",
             "classification_5","homeowner_desc","kid_category_desc","has_demographics"]
out = e10.merge(e11[["household_key","snapshot_day"]+demo_cols], on=["household_key","snapshot_day"], how="left")
out["has_demographics"] = out["has_demographics"].fillna(0).astype(int)
print(out.shape, "| demo cols:", [c for c in out.columns if c in demo_cols])
p = A.save_table(out, "e014_demo_l13fix.parquet")
print(p)
