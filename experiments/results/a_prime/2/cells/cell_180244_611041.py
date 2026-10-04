import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

TRAIN_DAYS = agent_api.snapshot_days()['train']; VAL_DAYS = agent_api.snapshot_days()['validation']
T = agent_api.train_targets()

def matrix(df_tr, df_va, feat_cols):
    Xtr_blocks, Xva_blocks, names = [], [], []
    for c in feat_cols:
        s_tr = df_tr[c].astype(object); s_va = df_va[c].astype(object)
        if s_tr.map(lambda v: isinstance(v, str)).all():
            cats = list(pd.Series(s_tr.dropna().unique()).sort_values())
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
            a_tr = np.nan_to_num(a_tr, nan=med[0]); a_va = np.nan_to_num(a_va, nan=med[0])
            Xtr_blocks.append(a_tr); Xva_blocks.append(a_va); names.append(c)
    Xtr = np.hstack(Xtr_blocks); Xva = np.hstack(Xva_blocks)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
    Xtr = np.clip((Xtr - mu) / sd, -5, 5); Xva = np.clip((Xva - mu) / sd, -5, 5)
    return Xtr, Xva, names

def ridge_eval(table, feat_cols=None, tag='', lam_grid=(1,10,100,1000)):
    df = table.merge(T, on=['household_key','snapshot_day'], how='inner')
    if feat_cols is None:
        feat_cols = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    tr = df[df.snapshot_day.isin(TRAIN_DAYS)]; va = df[df.snapshot_day.isin(VAL_DAYS)]
    fit_df = tr[tr.snapshot_day <= 375]; tune_df = tr[tr.snapshot_day.isin([403,431])]
    Xf, Xt, _ = matrix(fit_df, tune_df, feat_cols)
    yf = fit_df.future_spend_4w.values.astype(np.float64); yt = tune_df.future_spend_4w.values.astype(np.float64)
    Xf1 = np.hstack([Xf, np.ones((len(Xf),1),dtype=np.float32)]); Xt1 = np.hstack([Xt, np.ones((len(Xt),1),dtype=np.float32)])
    best = None
    for lam in lam_grid:
        A = Xf1.T @ Xf1; A[np.diag_indices_from(A)] += lam; A[-1,-1] -= lam
        w = np.linalg.solve(A, Xf1.T @ yf)
        m = np.abs(Xt1 @ w - yt).mean()
        if best is None or m < best[1]: best = (lam, m)
    lam = best[0]
    Xtr, Xva, names = matrix(tr, va, feat_cols)
    ytr = tr.future_spend_4w.values.astype(np.float64); yva = va.future_spend_4w.values.astype(np.float64)
    Xtr1 = np.hstack([Xtr, np.ones((len(Xtr),1),dtype=np.float32)]); Xva1 = np.hstack([Xva, np.ones((len(Xva),1),dtype=np.float32)])
    A = Xtr1.T @ Xtr1; A[np.diag_indices_from(A)] += lam; A[-1,-1] -= lam
    w = np.linalg.solve(A, Xtr1.T @ ytr)
    pred = Xva1 @ w
    mae = np.abs(pred - yva).mean()
    ss = ((yva - yva.mean())**2).sum(); r2 = 1 - ((pred - yva)**2).sum()/ss
    print(f'{tag:28s} lam={lam:5d} nfeat={Xtr.shape[1]:4d} OFFLINE val MAE={mae:8.3f} R2={r2:6.4f}')
    return mae, pred, va[['household_key','snapshot_day']].assign(pred=pred)

base = agent_api.baseline_features()
rec = agent_api.load_saved('recency_agg.parquet')
e9 = agent_api.load_saved('e009_ewma_longlags.parquet')
e1 = base.merge(rec, on=['household_key','snapshot_day'], how='inner')
print('shapes: base', base.shape, 'e1', e1.shape, 'e9', e9.shape)
m0,_,_ = ridge_eval(base, tag='E000 (harness 92.446)')
m1,_,_ = ridge_eval(e1, tag='E001 (harness 64.043)')
m9,p9,va9 = ridge_eval(e9, tag='E009 (harness 62.292)')
