import agent_api, pandas as pd, numpy as np
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
tt = agent_api.train_targets()
t431 = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w.reindex(E431.index)

def num(df):
    return df.drop(columns=["snapshot_day"]).apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

Xall = num(E431).values
y = t431.values
b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xall)), Xall]), y, rcond=None)
pred = np.column_stack([np.ones(len(Xall)), Xall])@b
print("in-sample R2 all E011 feats at 431:", round(1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum(),3))
print("in-sample MAE:", round(np.abs(y-pred).mean(),2))

tr_days = [d for d in sorted(E.snapshot_day.unique()) if d<431]
trX = E[E.snapshot_day.isin(tr_days)]
Xtr = num(trX).values
ytr = tt.set_index(["snapshot_day","household_key"]).loc[list(zip(trX.snapshot_day.astype(int), trX.household_key))].values
b2, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)
pred2 = np.column_stack([np.ones(len(Xall)), Xall])@b2
print("holdout-431 MAE (linear, all E011):", round(np.abs(y-pred2).mean(),2))

sp = ["spend_28","spend_84","spend_180","spend_365","ew_28","ew_84","ew_180","spend_28_prior","spend_84_prior","spend_lag364","lr_mean28","avg_basket_84","baskets_28","baskets_84","trips_per_wk_84","spend_364","rd_84","rd_364","spend_7","spend_14"]
Xs = E431[sp].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values
Xtr_s = trX[sp].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values
b3, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr_s)), Xtr_s]), ytr, rcond=None)
pred3 = np.column_stack([np.ones(len(Xs)), Xs])@b3
print("holdout-431 MAE (linear, spend-only subset):", round(np.abs(y-pred3).mean(),2))
