
import agent_api, pandas as pd, numpy as np

sd = agent_api.snapshot_days()

def prep_table(t, drop_extra=()):
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and c not in drop_extra]
    def prep(df):
        X = df[cols].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        return X.astype(np.float64)
    return cols, prep

def ridge_svd(X, y, lam):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    b = U.T @ y
    w = Vt.T @ (b * s / (s**2 + lam))
    return w

def evaluate(t, name, drop_extra=()):
    tt = agent_api.train_targets()
    m = t.merge(tt, on=['household_key','snapshot_day'])
    tr = m[m.snapshot_day.isin(sd['train'])]
    cols, prep = prep_table(t, drop_extra)
    Xtr_df = prep(tr)
    keep = Xtr_df.columns[Xtr_df.notna().any()].tolist()
    Xtr = Xtr_df[keep].values
    ytr = tr['future_spend_4w'].values
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    mu, sg = Xtr.mean(0), Xtr.std(0)
    dead = sg < 1e-12
    Xtr_s = np.clip((Xtr-mu)/np.where(dead,1,sg), -8, 8)[:, ~dead]
    iv = (tr.snapshot_day==431).values
    out = {}
    for lam in [10, 100, 1000, 10000]:
        w = ridge_svd(Xtr_s[~iv], ytr[~iv], lam)
        inner = np.abs(Xtr_s[iv]@w - ytr[iv]).mean()
        w2 = ridge_svd(Xtr_s, ytr, lam)
        out[lam] = (round(inner,2), round(np.abs(Xtr_s@w2 - ytr).mean(),2))
    print(f'{name}: nfeat={Xtr_s.shape[1]}  (lam: inner431, fulltrain-trainMAE) = {out}')
    return out

# E000 baseline for calibration
b = agent_api.baseline_features()
evaluate(b, 'E000-baseline')
# E009
t9 = agent_api.load_saved('e009_ewma_longlags.parquet')
evaluate(t9, 'E009', drop_extra=('index',))
