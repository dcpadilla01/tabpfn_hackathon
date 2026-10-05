import agent_api, pandas as pd, numpy as np
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
tt = agent_api.train_targets()
tmap = tt.set_index(["snapshot_day","household_key"]).future_spend_4w
t431 = tmap.loc[(431, E431.index)].values

def num(df):
    return df.drop(columns=["snapshot_day"]).apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

tr_days = [d for d in sorted(E.snapshot_day.unique()) if d<431]
trX = E[E.snapshot_day.isin(tr_days)]
Xtr = num(trX).values
ytr = tmap.loc[list(zip(trX.snapshot_day.astype(int), trX.household_key))].values

def fit_eval(cols_tr, cols_t, name):
    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(cols_tr)), cols_tr]), ytr, rcond=None)
    pred = np.column_stack([np.ones(len(cols_t)), cols_t])@b
    print(name, "holdout-431 MAE:", round(np.abs(y-pred).mean(),2))

fit_eval(Xtr, num(E431).values, "all-E011")
sp = ["spend_28","spend_84","spend_180","spend_365","ew_28","ew_84","ew_180","spend_28_prior","spend_84_prior","spend_lag364","lr_mean28","avg_basket_84","baskets_28","baskets_84","trips_per_wk_84","spend_364","rd_84","rd_364","spend_7","spend_14"]
fit_eval(trX[sp].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values,
         E431[sp].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values, "spend-only")
print("val rows in E011:", (E.snapshot_day>=459).sum())
