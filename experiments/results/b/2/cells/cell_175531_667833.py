import pandas as pd, numpy as np, agent_api as A

train_days = A.snapshot_days()['train']
tt = A.train_targets()

def prep(path):
    t = A.load_saved(path)
    feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
    return df, feats

def cv_mae(df, feats, lam=30.0, clip=None):
    X = df[feats].astype(float).values.copy()
    if clip is not None:
        lo, hi = clip
        X = np.clip(X, lo, hi)
    y = df.future_spend_4w.values
    days = df.snapshot_day.values
    mu = np.nanmean(X,0); sd = np.nanstd(X,0)+1e-9
    X = np.where(np.isnan(X), mu, (X-mu)/sd)
    errs = []
    for s in train_days:
        m = days != s
        Xt, yt = X[m], y[m]
        b = np.linalg.solve(Xt.T@Xt + lam*np.eye(X.shape[1]), Xt.T@yt)
        pv = np.clip(X[~m]@b, 0, None)
        errs.append(np.abs(pv - y[~m]).mean())
    return float(np.mean(errs)), errs

for path in ['e015_market_ctx.parquet','e018_hinge_prune.parquet','e019_final.parquet']:
    df, feats = prep(path)
    for lam in [30,100]:
        m,_ = cv_mae(df, feats, lam)
        print(f"{path:32s} lam={lam:4d} CV={m:.3f} (nfeat={len(feats)})")
