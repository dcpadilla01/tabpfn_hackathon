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
