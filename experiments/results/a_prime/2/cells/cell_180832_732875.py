import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
T = agent_api.train_targets()

def matrix(df_tr, df_va, feat_cols):
    Xtr_blocks, Xva_blocks, names = [], [], []
    for c in feat_cols:
        s_tr = df_tr[c].astype(object); s_va = df_va[c].astype(object)
        nonan = s_tr.dropna()
        is_str = len(nonan) > 0 and nonan.map(lambda v: isinstance(v, str)).all()
        if is_str:
            cats = list(pd.Series(nonan.unique()).sort_values())
            mapping = {v: i for i, v in enumerate(cats)}
            def oh(s):
                codes = s.map(mapping).astype(float).fillna(-1).values.astype(int)
                M = np.zeros((len(s), len(cats)), dtype=np.float32)
                ok = codes >= 0
                M[np.arange(len(s))[ok], codes[ok]] = 1.0
                return M
            Xtr_blocks.append(oh(s_tr)); Xva_blocks.append(oh(s_va)); names += [f'{c}__{v}' for v in cats]
        else:
            a_tr = pd.to_numeric(s_tr, errors='coerce').astype(np.float32).values.reshape(-1,1).copy()
            a_va = pd.to_numeric(s_va, errors='coerce').astype(np.float32).values.reshape(-1,1).copy()
            med = np.nanmedian(a_tr, axis=0)
            if not np.isfinite(med[0]): med[0] = 0.0
            a_tr = np.nan_to_num(a_tr, nan=med[0]); a_va = np.nan_to_num(a_va, nan=med[0])
            Xtr_blocks.append(a_tr); Xva_blocks.append(a_va); names.append(c)
    Xtr = np.hstack(Xtr_blocks); Xva = np.hstack(Xva_blocks)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
    Xtr = np.clip((Xtr - mu) / sd, -5, 5); Xva = np.clip((Xva - mu) / sd, -5, 5)
    return Xtr, Xva, names

def ridge_eval(table, feat_cols=None, tag='', lam_grid=(1000,), fit_max=347, tune_days=(375,403,431), scale=1.0):
    df = table.merge(T, on=['household_key','snapshot_day'], how='inner')
    if feat_cols is None:
        feat_cols = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    fit_df = df[df.snapshot_day <= fit_max]; tune_df = df[df.snapshot_day.isin(list(tune_days))]
    Xf, Xt, _ = matrix(fit_df, tune_df, feat_cols)
    Xf = np.clip(Xf*scale, -5, 5); Xt = np.clip(Xt*scale, -5, 5)
    yf = fit_df.future_spend_4w.values.astype(np.float64); yt = tune_df.future_spend_4w.values.astype(np.float64)
    Xf1 = np.hstack([Xf, np.ones((len(Xf),1),dtype=np.float32)]); Xt1 = np.hstack([Xt, np.ones((len(Xt),1),dtype=np.float32)])
    best = None
    for lam in lam_grid:
        A = Xf1.T @ Xf1; A[np.diag_indices_from(A)] += lam; A[-1,-1] -= lam
        w = np.linalg.solve(A, Xf1.T @ yf)
        m = np.abs(Xt1 @ w - yt).mean()
        if best is None or m < best[1]: best = (lam, m)
    lam, mae = best
    print(f'{tag:40s} lam={lam:5d} nfeat={Xf.shape[1]:4d} INTERNAL MAE={mae:8.3f}')
    return mae

e9 = agent_api.load_saved('e009_ewma_longlags.parquet')
feat_cols = [c for c in e9.columns if c not in ('household_key','snapshot_day')]
# identify heavy-tailed nonneg columns
num = e9[feat_cols].select_dtypes(include=[np.number])
heavy = [c for c in num.columns if e9[c].min() >= 0 and e9[c].quantile(0.99) > 50]
print('heavy-tailed cols:', len(heavy), heavy[:15])
e9l = e9.copy()
for c in heavy:
    e9l[c] = np.log1p(e9l[c].astype(np.float64))
ridge_eval(e9,  tag='e9 raw            scale=1.0')
ridge_eval(e9l, tag='e9 log-heavy      scale=1.0')
ridge_eval(e9l, tag='e9 log-heavy      scale=0.3')
ridge_eval(e9l, tag='e9 log-heavy      scale=0.1')
# sqrt variant
e9s = e9.copy()
for c in heavy:
    e9s[c] = np.sqrt(e9s[c].astype(np.float64))
ridge_eval(e9s, tag='e9 sqrt-heavy     scale=0.3')