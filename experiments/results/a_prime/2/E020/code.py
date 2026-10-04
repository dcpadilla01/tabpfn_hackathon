import agent_api, pandas as pd, numpy as np
b = agent_api.load_saved('e009_ewma_longlags.parquet')
print('shape', b.shape)
print(b['snapshot_day'].value_counts().sort_index())
cols = list(b.columns)
for pat in ['spend','tlag','ewma','trip','index']:
    print(pat, [c for c in cols if pat in c.lower()][:40])
print('dtypes', b.dtypes.value_counts().to_dict())
t = agent_api.train_targets()
print(t.shape, t.columns.tolist())
print(t.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']))


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

TRAIN_DAYS = agent_api.snapshot_days()['train']; VAL_DAYS = agent_api.snapshot_days()['validation']
T = agent_api.train_targets()

def matrix(df_tr, df_va, feat_cols):
    Xtr_blocks, Xva_blocks, names = [], [], []
    for c in feat_cols:
        s_tr, s_va = df_tr[c], df_va[c]
        if str(s_tr.dtype) == 'category' or s_tr.dtype == object:
            cats = list(pd.Series(s_tr.dropna().unique()).sort_values())
            mapping = {v: i for i, v in enumerate(cats)}
            def oh(s):
                codes = s.map(mapping).fillna(-1).astype(int).values
                M = np.zeros((len(s), len(cats)), dtype=np.float32)
                ok = codes >= 0
                M[np.arange(len(s))[ok], codes[ok]] = 1.0
                return M
            Xtr_blocks.append(oh(s_tr)); Xva_blocks.append(oh(s_va)); names += [f'{c}__{v}' for v in cats]
        else:
            a_tr = s_tr.astype(np.float32).values.reshape(-1,1).copy()
            a_va = s_va.astype(np.float32).values.reshape(-1,1).copy()
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


# ---- cell ----
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


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
T = agent_api.train_targets()
base = agent_api.baseline_features()
df = base.merge(T, on=['household_key','snapshot_day'], how='inner')
tr = df[df.snapshot_day <= 375]
print(base.dtypes)
for c in base.columns:
    if c in ('household_key','snapshot_day'): continue
    s = pd.to_numeric(base[c].astype(object), errors='coerce')
    print(c, base[c].dtype, 'nan:', s.isna().sum(), 'max:', np.nanmax(np.abs(s.values.astype(np.float64))) if s.notna().any() else None)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

TRAIN_DAYS = agent_api.snapshot_days()['train']; VAL_DAYS = agent_api.snapshot_days()['validation']
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
m0,_,_ = ridge_eval(base, tag='E000 (harness 92.446)')
m1,_,_ = ridge_eval(e1, tag='E001 (harness 64.043)')
m9,p9,va9 = ridge_eval(e9, tag='E009 (harness 62.292)')


# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

TRAIN_DAYS = agent_api.snapshot_days()['train']
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

def ridge_eval(table, feat_cols=None, tag='', lam_grid=(1,10,100,1000), fit_max=347, tune_days=(375,403,431)):
    df = table.merge(T, on=['household_key','snapshot_day'], how='inner')
    if feat_cols is None:
        feat_cols = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    fit_df = df[df.snapshot_day <= fit_max]; tune_df = df[df.snapshot_day.isin(list(tune_days))]
    Xf, Xt, _ = matrix(fit_df, tune_df, feat_cols)
    yf = fit_df.future_spend_4w.values.astype(np.float64); yt = tune_df.future_spend_4w.values.astype(np.float64)
    Xf1 = np.hstack([Xf, np.ones((len(Xf),1),dtype=np.float32)]); Xt1 = np.hstack([Xt, np.ones((len(Xt),1),dtype=np.float32)])
    best = None
    for lam in lam_grid:
        A = Xf1.T @ Xf1; A[np.diag_indices_from(A)] += lam; A[-1,-1] -= lam
        w = np.linalg.solve(A, Xf1.T @ yf)
        m = np.abs(Xt1 @ w - yt).mean()
        if best is None or m < best[1]: best = (lam, m)
    lam, mae = best
    print(f'{tag:34s} lam={lam:5d} nfeat={Xf.shape[1]:4d} INTERNAL MAE(375/403/431)={mae:8.3f}')
    return mae, lam

base = agent_api.baseline_features()
rec = agent_api.load_saved('recency_agg.parquet')
e9 = agent_api.load_saved('e009_ewma_longlags.parquet')
e1 = base.merge(rec, on=['household_key','snapshot_day'], how='inner')
m0,_ = ridge_eval(base, tag='E000 (harness 92.446)')
m1,_ = ridge_eval(e1, tag='E001 (harness 64.043)')
m9,_ = ridge_eval(e9, tag='E009 (harness 62.292)')

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
TRAIN_DAYS = agent_api.snapshot_days()['train']
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

def ridge_eval(table, feat_cols=None, tag='', lam_grid=(1,10,100,1000), fit_max=347, tune_days=(375,403,431)):
    df = table.merge(T, on=['household_key','snapshot_day'], how='inner')
    if feat_cols is None:
        feat_cols = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    fit_df = df[df.snapshot_day <= fit_max]; tune_df = df[df.snapshot_day.isin(list(tune_days))]
    Xf, Xt, _ = matrix(fit_df, tune_df, feat_cols)
    yf = fit_df.future_spend_4w.values.astype(np.float64); yt = tune_df.future_spend_4w.values.astype(np.float64)
    Xf1 = np.hstack([Xf, np.ones((len(Xf),1),dtype=np.float32)]); Xt1 = np.hstack([Xt, np.ones((len(Xt),1),dtype=np.float32)])
    best = None
    for lam in lam_grid:
        A = Xf1.T @ Xf1; A[np.diag_indices_from(A)] += lam; A[-1,-1] -= lam
        w = np.linalg.solve(A, Xf1.T @ yf)
        m = np.abs(Xt1 @ w - yt).mean()
        if best is None or m < best[1]: best = (lam, m)
    lam, mae = best
    print(f'{tag:36s} lam={lam:5d} nfeat={Xf.shape[1]:4d} INTERNAL MAE={mae:8.3f}')
    return mae

e9 = agent_api.load_saved('e009_ewma_longlags.parquet')
for name in ['e019_curated.parquet','e019_curated2.parquet','e019_curated3.parquet','e011_pruned.parquet']:
    t = agent_api.load_saved(name)
    ridge_eval(t, tag=name)
ridge_eval(e9, tag='e9 lam(30,100,300)', lam_grid=(30,100,300))
ridge_eval(e9, tag='e9 lam(3000,10000)', lam_grid=(3000,10000))

# ---- cell ----
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

def ridge_eval(table, feat_cols=None, tag='', lam_grid=(1,10,100,1000), fit_max=347, tune_days=(375,403,431), scale=1.0, clip=5.0):
    df = table.merge(T, on=['household_key','snapshot_day'], how='inner')
    if feat_cols is None:
        feat_cols = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    fit_df = df[df.snapshot_day <= fit_max]; tune_df = df[df.snapshot_day.isin(list(tune_days))]
    Xf, Xt, _ = matrix(fit_df, tune_df, feat_cols)
    Xf = np.clip(Xf*scale, -clip, clip); Xt = np.clip(Xt*scale, -clip, clip)
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
e11 = agent_api.load_saved('e011_pruned.parquet')
for a in (0.5, 0.3, 0.2, 0.1):
    ridge_eval(e9, tag=f'e9 scale={a}', lam_grid=(1000,), scale=a)
for a in (0.5, 0.3, 0.2):
    ridge_eval(e11, tag=f'e011pruned scale={a}', lam_grid=(1000,), scale=a)

# ---- cell ----
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

# ---- cell ----
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
    print(f'{tag:44s} lam={lam:5d} nfeat={Xf.shape[1]:4d} INTERNAL MAE={mae:8.3f}')
    return mae

e9 = agent_api.load_saved('e009_ewma_longlags.parquet')
top = ['spend_28','tlag_mean','ewma_4','spend_56','spend_84','ts_ewma4','tlag_2','tlag_3','spend_112','ewma_8']
e9d2 = e9.copy()
for c in top: e9d2[c+'_dup'] = e9d2[c]
ridge_eval(e9d2, tag='e9 +dup(top10)x1 scale=0.3', scale=0.3)
e9d3 = e9.copy()
for c in top[:3]: e9d3[c+'_d1'] = e9d3[c]; e9d3[c+'_d2'] = e9d3[c]
ridge_eval(e9d3, tag='e9 +dup(top3)x2 scale=0.3', scale=0.3)
ridge_eval(e9d2, tag='e9 +dup(top10)x1 scale=1.0', scale=1.0)

# ---- cell ----
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e9 = agent_api.load_saved('e009_ewma_longlags.parquet')
out = e9.copy()
for c in out.columns:
    if c in ('household_key','snapshot_day'): continue
    if pd.api.types.is_numeric_dtype(out[c]):
        out[c] = (out[c].astype(np.float64) * 0.3)
print(out.shape, 'scaled numeric cols:', sum(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns))
path = agent_api.save_table(out, 'e020_scaled03.parquet')
print(path)
chk = agent_api.load_saved('e020_scaled03.parquet')
print('reload shape', chk.shape, 'spend_28 head:', chk['spend_28'].head(3).tolist())
print('snapshot days:', sorted(chk.snapshot_day.unique()))
