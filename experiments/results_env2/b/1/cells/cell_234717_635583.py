import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()

def prep(df):
    df = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
    y = df['future_spend_4w'].irreducible
    return df

def ridge_on_cols(df, cols, fit_days, ev_days, alpha):
    d = tt.merge(df[["household_key","snapshot_day"]+cols], on=["household_key","snapshot_day"], how="inner")
    y = d['future_spend_4w'].values.astype(float)
    X = d[cols].astype(float).fillna(0).values
    f = np.isin(d['snapshot_day'], fit_days); e = np.isin(d['snapshot_day'], ev_days)
    mu=X[f].mean(0); sd=X[f].std(0); sd[sd==0]=1
    Z=(X-mu)/sd
    A=Z[f].T@Z[f]+alpha*np.eye(len(cols)); b=Z[f].T@y[f]
    w=np.linalg.solve(A,b)
    p=Z[e]@w
    return np.mean(np.abs(p-y[e]))

e8 = agent_api.load_saved("e008_level_shape.parquet")
fit=list(range(95,348,28)); ev=[375,403,431]
num = [c for c in e8.columns if c not in ("household_key","snapshot_day") and pd.api.types.is_numeric_dtype(e8[c])]

# single-feature probes at big alpha
res=[]
for c in num:
    try:
        m = ridge_on_cols(e8, [c], fit, ev, 30000)
        res.append((m,c))
    except Exception as ex:
        pass
res.sort()
print("best single numeric features (local ridge):")
for m,c in res[:15]: print(round(m,2), c)
print("...")
for m,c in res[-5:]: print(round(m,2), c)
