import pandas as pd, numpy as np, agent_api

def ridge_cv(path_or_df, val_days=(403,431), alphas=(30.,100.,300.,1000.)):
    df = agent_api.load_saved(path_or_df) if isinstance(path_or_df,str) else path_or_df
    tt = agent_api.train_targets()
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    y = m['future_spend_4w'].astype(float).values
    sd = m['snapshot_day'].values
    X = pd.get_dummies(m.drop(columns=['future_spend_4w']), dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan).values
    maes = {}
    for a in alphas:
        errs = []
        for vd in val_days:
            tr = sd < vd; va = sd == vd
            mu = np.nanmean(X[tr],0); sg = np.nanstd(X[tr],0)+1e-9
            mu = np.where(np.isnan(mu), 0., mu); sg = np.where(np.isnan(sg), 1., sg)
            Z = np.where(np.isnan(X[tr]), mu, X[tr]); Z = (Z-mu)/sg
            Zv = np.where(np.isnan(X[va]), mu, X[va]); Zv = (Zv-mu)/sg
            w = np.linalg.solve(Z.T@Z + a*np.eye(Z.shape[1]), Z.T@(y[tr]-y[tr].mean()))
            p = Zv@w + y[tr].mean()
            errs.append(np.abs(p - y[va]).mean())
        maes[a] = np.mean(errs)
    b = min(maes, key=maes.get)
    return b, maes

for name, path in [('E000','BASELINE'), ('E001','e001_recent_behavior.parquet'),
                   ('E004','e004_temporal.parquet'), ('E005','e005_decay_gapcv.parquet'),
                   ('E010','e010_lifecycle.parquet'), ('E011','e011_discounts.parquet'),
                   ('E012','e012_hh_target_enc.parquet'), ('E014','e014_transforms.parquet'),
                   ('E015','e015_best_pseudo.parquet')]:
    df = agent_api.baseline_features() if path=='BASELINE' else path
    b, maes = ridge_cv(df)
    print(f'{name}: alpha={b:.0f} cvMAE={maes[b]:.3f}  all={ {k:round(v,2) for k,v in maes.items()} }')
