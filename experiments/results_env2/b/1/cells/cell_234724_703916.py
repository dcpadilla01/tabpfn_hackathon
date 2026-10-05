import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
e8 = agent_api.load_saved("e008_level_shape.parquet")

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

fit=list(range(95,348,28)); ev=[375,403,431]
# small curated sets
sets = {
 "sp728 only": ["sp728"],
 "sp364+728": ["sp364","sp728"],
 "top5": ["sp728","sp364","sp364_rate","prods728","nact728"],
 "top10": ["sp728","sp364","sp364_rate","prods728","nact728","rdisc364","nact364","trips728","trips364","z_max4w_hist"],
 "sp84+sp728": ["sp84","sp728"],
 "sp84+sp728+recency": ["sp84","sp728","days_since_last"],
 "sp84+sp728+med4w": ["sp84","sp728","z_med4w_hist"],
 "sp28+84+728+med4w": ["sp28","sp84","sp728","z_med4w_hist"],
}
for k,v in sets.items():
    print(k, round(ridge_on_cols(e8, v, fit, ev, 30000),3))
