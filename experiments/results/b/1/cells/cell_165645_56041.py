import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def local_eval(df, alpha_grid=(0.3,1,3,10,30,100)):
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    feat = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    Xn = M[num_cols].astype(float).copy()
    tr_mask_full = (M.snapshot_day <= 375).values
    med = Xn[tr_mask_full].median()
    Xn = Xn.fillna(med).fillna(0.0)
    mu = Xn[tr_mask_full].mean(); sd = Xn[tr_mask_full].std().replace(0,1)
    Xn = ((Xn - mu) / sd).values
    cat_mats = []
    for c in cat_cols:
        d = pd.get_dummies(M[c].astype(str), prefix=c, dummy_na=True)
        cat_mats.append(d.values.astype(float))
    X = np.hstack([Xn] + cat_mats) if cat_mats else Xn
    inner_tr = (M.snapshot_day <= 347).values
    inner_va = (M.snapshot_day == 375).values
    va = (M.snapshot_day >= 403).values
    y = M.future_spend_4w.values
    def fit(Xs, ys, a):
        return np.linalg.solve(Xs.T @ Xs + a*np.eye(Xs.shape[1]), Xs.T @ ys)
    best = (None, 1e18)
    for a in alpha_grid:
        b = fit(X[inner_tr], y[inner_tr], a)
        p = np.clip(X[inner_va] @ b, -20, 20)
        m = np.mean(np.abs(p - y[inner_va]))
        if m < best[1]: best = (a, m)
    a = best[0]
    b = fit(X[tr_mask_full], y[tr_mask_full], a)
    p = np.clip(X[va] @ b, -20, 20)
    return np.mean(np.abs(p - y[va])), a, X.shape[1]

harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}
res = {}
for nm in harness:
    df = A.load_saved(nm + ".parquet")
    m, a, nf = local_eval(df)
    res[nm] = (m, harness[nm])
    print(f"{nm:22s} local={m:7.3f} (a={a}, k={nf}) harness={harness[nm]:7.3f}")
lv = pd.Series([res[n][0] for n in harness]); hv = pd.Series([res[n][1] for n in harness])
print("rank corr:", lv.rank().corr(hv.rank()), "| pearson:", lv.corr(hv))