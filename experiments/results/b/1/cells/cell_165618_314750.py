import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def dedup(df):
    return df.loc[:, ~pd.Index(df.columns).duplicated()]

def design(df, num_cols, cat_cols, tr_mask):
    Xn = df[num_cols].astype(float).copy()
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0)
    mu = Xn[tr_mask].mean(); sd = Xn[tr_mask].std().replace(0,1)
    Xn = (Xn - mu) / sd
    mats = [Xn.values]
    for c in cat_cols:
        d = pd.get_dummies(df[c].astype(str), prefix=c, dummy_na=True)
        d = d.reindex(columns=cat_levels[c], fill_value=0)
        mats.append(d.values.astype(float))
    return np.hstack(mats)

def fit_ridge(X, y, alpha):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)

def local_eval(df, alpha_grid=(0.3,1,3,10,30,100), log_target=False):
    df = dedup(df)
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    feat = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    global cat_levels
    cat_levels = {}
    for c in cat_cols:
        cat_levels[c] = list(pd.get_dummies(M[c].astype(str), dummy_na=True).columns)
    inner_tr = (M.snapshot_day <= 347).values
    inner_va = (M.snapshot_day == 375).values
    tr = (M.snapshot_day <= 375).values
    va = (M.snapshot_day >= 403).values
    Xi = design(M, num_cols, cat_cols, inner_tr)
    Xtr = design(M, num_cols, cat_cols, tr)
    yi, yvi = M.future_spend_4w.values[inner_tr], M.future_spend_4w.values[inner_va]
    ytr, yva = M.future_spend_4w.values[tr], M.future_spend_4w.values[va]
    f = (np.log1p if log_target else (lambda z: z))
    finv = (np.expm1 if log_target else (lambda z: z))
    best = (None, 1e18)
    for a in alpha_grid:
        b = fit_ridge(Xi[inner_tr], f(yi), a)
        p = finv(np.clip(Xi[inner_va] @ b, -20, 20))
        m = np.mean(np.abs(p - yvi))
        if m < best[1]: best = (a, m)
    a = best[0]
    b = fit_ridge(Xtr[tr], f(ytr), a)
    p = finv(np.clip(Xtr[va] @ b, -20, 20))
    return np.mean(np.abs(p - yva)), a, Xi.shape[1]

harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}
names = list(harness.keys())
res = {}
for nm in names:
    df = A.load_saved(nm + ".parquet")
    m, a, nf = local_eval(df)
    res[nm] = (m, a, nf, harness[nm])
    print(f"{nm:22s} local={m:7.3f} (a={a}, k={nf}) harness={harness[nm]:7.3f}")
lv = pd.Series([res[n][0] for n in names]); hv = pd.Series([res[n][3] for n in names])
print("rank corr:", lv.rank().corr(hv.rank()), "| pearson:", lv.corr(hv))