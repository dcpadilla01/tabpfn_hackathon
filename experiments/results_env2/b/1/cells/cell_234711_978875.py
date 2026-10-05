import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()

def prep(df):
    df = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
    y = df['future_spend_4w'].values.astype(float)
    cats = [c for c in df.columns if c not in ("household_key","snapshot_day","future_sp4w_dummy") and not pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(df[c])]
    Xn = df[nums].astype(float).fillna(0).values
    Xc = pd.get_dummies(df[cats].astype(str), dummy_na=True).values.astype(float) if cats else np.zeros((len(df),40))
    X = np.column_stack([Xn, Xc]) if Xc.size else Xn
    return X, y, df['snapshot_day'].values

def local_eval(path, fit_days, ev_days, alpha=300.0):
    df = agent_api.load_saved(path)
    X, y, day = prep(df)
    f = np.isin(day, fit_days); e = np.isin(day, ev_days)
    mu = X[f].mean(0); sd = X[f].std(0); sd[sd==0]=1
    Z=(X-mu)/sd
    A=Z[f].T@Z[f]+alpha*np.eye(Z.shape[1]); b=Z[f].T@y[f]
    w=np.linalg.solve(A,b)
    p=Z[e]@w
    return np.mean(np.abs(p-y[e]))

# try larger alphas; ridge underfitting suggests alpha too small or features too noisy
fit=list(range(95,348,28)); ev=[375,403,431]
for a in [300, 1000, 3000, 10000, 30000]:
    print("e008 alpha",a, round(local_eval("e008_level_shape.parquet", fit, ev, a),3))
