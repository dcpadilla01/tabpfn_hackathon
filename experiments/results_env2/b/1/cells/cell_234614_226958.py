import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

tt = agent_api.train_targets()

def prep(df):
    df = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
    y = df['future_spend_4w'].values.astype(float)
    cats = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w") and pd.api.types.is_numeric_dtype(df[c])]
    Xn = df[nums].astype(float).fillna(0).values
    Xc = pd.get_dummies(df[cats].astype(str), dummy_na=True).values.astype(float) if cats else np.zeros((len(df),0))
    X = np.column_stack([Xn, Xc]) if Xc.size else Xn
    return X, y, df['snapshot_day'].values

def ridge_fit(X, y, alpha):
    mu = X.mean(0); sd = X.std(0); sd[sd==0]=1
    Z = (X-mu)/sd
    A = Z.T@Z + alpha*np.eye(Z.shape[1])
    b = Z.T@y
    w = np.linalg.solve(A, b)
    return w, mu, sd

def ridge_pred(X, model):
    w, mu, sd = model
    return (X-mu)/sd @ w

def local_eval(path, fit_days, ev_days, alpha=100.0):
    df = agent_api.load_saved(path)
    X, y, day = prep(df)
    f = np.isin(day, fit_days); e = np.isin(day, ev_days)
    m = ridge_fit(X[f], y[f], alpha)
    p = ridge_pred(X[e], m)
    return np.mean(np.abs(p - y[e]))

print("alpha check on e008 (fit 95-347, ev 375/403/431):")
for a in [1,10,30,100,300,1000]:
    print(a, round(local_eval("e008_level_shape.parquet", list(range(95,348,28)), [375,403,431], a),3))
