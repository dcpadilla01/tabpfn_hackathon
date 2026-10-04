import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}

def prep(M, tr_mask):
    feat = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    Xn = M[num_cols].astype(float).copy()
    # compress heavy tails: sign(x)*log1p(|x|)
    Xn = np.sign(Xn) * np.log1p(np.abs(Xn))
    Xn = pd.DataFrame(Xn, columns=num_cols)
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0)
    mu = Xn[tr_mask].mean(); sd = Xn[tr_mask].std().replace(0,1)
    Xn = ((Xn - mu)/sd).values
    mats = [Xn]
    for c in cat_cols:
        d = pd.get_dummies(M[c].astype(str), prefix=c, dummy_na=True)
        mats.append(d.values.astype(float))
    return np.hstack(mats)

def local_eval(df):
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    trm = (M.snapshot_day <= 375).values
    X = prep(M, trm)
    inner_tr = (M.snapshot_day <= 347).values
    inner_va = (M.snapshot_day == 375).values
    va = (M.snapshot_day >= 403).values
    y = M.future_spend_4w.values
    ly = np.log1p(y)
    def fit(Xs, ys, a):
        return np.linalg.solve(Xs.T @ Xs + a*np.eye(Xs.shape[1]), Xs.T @ ys)
    best = (None, 1e18)
    for a in (0.3,1,3,10,30,100,300):
        b = fit(X[inner_tr], ly[inner_tr], a)
        p = np.expm1(np.clip(X[inner_va] @ b, -5, 5))
        m = np.mean(np.abs(p - y[inner_va]))
        if m < best[1]: best = (a, m)
    a = best[0]
    b = fit(X[trm], ly[trm], a)
    p = np.expm1(np.clip(X[va] @ b, -5, 5))
    return np.mean(np.abs(p - y[va])), a

res = {}
for nm in harness:
    df = A.load_saved(nm + ".parquet")
    m, a = local_eval(df)
    res[nm] = m
    print(f"{nm:22s} local={m:7.3f} (a={a}) harness={harness[nm]:7.3f}")
lv = pd.Series({n:res[n] for n in harness}); hv = pd.Series(harness)
print("rank corr:", lv.rank().corr(hv.rank()), "| pearson:", lv.corr(hv))