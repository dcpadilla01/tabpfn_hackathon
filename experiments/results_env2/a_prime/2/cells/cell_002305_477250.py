import pandas as pd, numpy as np, agent_api

def prep(df):
    tt = agent_api.train_targets()
    m = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    y = m['future_spend_4w'].astype(float).values
    X = pd.get_dummies(m.drop(columns=['future_spend_4w']), dummy_na=True).astype(float)
    X = X.replace([np.inf,-np.inf], np.nan)
    return X, y, m['snapshot_day'].values

for name, path in [('E011','e011_discounts.parquet'), ('E015','e015_best_pseudo.parquet')]:
    X, y, sd = prep(agent_api.load_saved(path))
    print(name, X.shape)
    # all-NaN columns overall
    allnan = X.columns[X.isna().all()].tolist()
    print('  all-NaN cols:', allnan)
    # all-NaN within any train fold
    for vd in (403,431):
        tr = sd < vd
        bad = [c for c in X.columns if X.loc[tr, c].isna().all()]
        if bad: print(f'  fold<{vd} all-NaN cols:', bad)
    # check inf
    print('  any inf:', np.isinf(X.values).sum())
