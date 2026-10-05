import pandas as pd, numpy as np, agent_api

TRAIN_DAYS = [95,123,151,179,207,235,263,291,319,347,375,403,431]
VAL_DAYS = [459,487,515,543]

def prep(df, tt=None):
    tt = agent_api.train_targets() if tt is None else tt
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    y = m['future_spend_4w'].astype(float).values
    # numeric + one-hot for object/bool
    X = pd.get_dummies(m.drop(columns=['future_spend_4w']), dummy_na=True)
    X = X.astype(float)
    return X.values, y, m['snapshot_day'].values

def ridge_cv(path_or_df, val_days=(403,431), alphas=(30.,100.,300.,1000.)):
    df = agent_api.load_saved(path_or_df) if isinstance(path_or_df,str) else path_or_df
    X, y, sd = prep(df)
    maes = {}
    for a in alphas:
        errs = []
        for vd in val_days:
            tr = sd < vd; va = sd == vd
            mu, sg = X[tr].mean(0), X[tr].std(0)+1e-9
            Z, Zv = (X[tr]-mu)/sg, (X[va]-mu)/sg
            w = np.linalg.solve(Z.T@Z + a*np.eye(Z.shape[1]), Z.T@(y[tr]-y[tr].mean()))
            p = Zv@w + y[tr].mean()
            errs.append(np.abs(p - y[va]).mean())
        maes[a] = np.mean(errs)
    best = min(maes, key=maes.get)
    return best, maes

for name, path in [('E000','BASELINE'), ('E001','e001_recent_behavior.parquet'),
                   ('E011','e011_discounts.parquet'), ('E012','e012_hh_target_enc.parquet'),
                   ('E015','e015_best_pseudo.parquet')]:
    df = agent_api.baseline_features() if path=='BASELINE' else path
    b, maes = ridge_cv(df)
    print(f'{name}: best_alpha={b:.0f} cvMAE={maes[b]:.3f}  all={ {k:round(v,2) for k,v in maes.items()} }')
