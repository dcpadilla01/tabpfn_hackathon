import agent_api, pandas as pd, numpy as np

E = agent_api.load_saved("e011_discounts.parquet")

spend_like = ["spend_7","spend_14","spend_28","spend_56","spend_84","spend_180","spend_365",
              "spend_28_prior","spend_84_prior","spend_lag336","spend_lag364","spend_lag392",
              "ew_7","ew_14","ew_28","ew_56","ew_84","ew_180","spend_364",
              "avg_basket_84","lr_mean28","lr_med28","rd_84","rd_364","cd_84","cd_364",
              "net_spend_364","basket_max_84","longrun_wk"]
count_like = ["baskets_28","baskets_84","trips_per_wk_84","trips_364","active_days_28",
              "active_days_364","n_products_84","n_stores_84","coup_trips_84","coup_trips_364",
              "redem_84","days_since_last","days_since_first","days_since_redem","active_28"]

new = {}
for c in spend_like + count_like:
    if c not in E.columns: continue
    x = pd.to_numeric(E[c], errors="coerce").clip(lower=0)
    new[c+"_sqrt"] = np.sqrt(x)
    if c in spend_like:
        new[c+"_log"] = np.log1p(x)
new = pd.DataFrame(new, index=E.index)
T = pd.concat([E, new], axis=1)
print("new feats:", new.shape[1], "total shape:", T.shape)

# quick holdout check at snapshot 431 (fit on train days <431, OLS)
try:
    tt = agent_api.train_targets()
    tmap = tt.set_index(["snapshot_day","household_key"]).future_spend_4w
    def num(df, cols):
        return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values
    feat_cols = [c for c in T.columns if c not in ("household_key","snapshot_day")]
    tr = T[T.snapshot_day < 431]; te = T[T.snapshot_day == 431]
    ytr = tmap.loc[list(zip(tr.snapshot_day.astype(int), tr.household_key))].values
    yte = tmap.loc[list(zip(te.snapshot_day.astype(int), te.household_key))].values
    for name, cols in [("E011 only", [c for c in E.columns if c not in ("household_key","snapshot_day")]),
                       ("E011+transforms", feat_cols)]:
        Xtr = num(tr, cols); Xte = num(te, cols)
        b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)
        pred = np.column_stack([np.ones(len(Xte)), Xte]) @ b
        print(name, "-> holdout-431 MAE:", round(np.abs(yte - pred).mean(), 3))
except Exception as e:
    print("check failed:", repr(e))

path = agent_api.save_table(T, "e014_transforms.parquet")
print("saved:", path)
