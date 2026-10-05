
import numpy as np, pandas as pd, xgboost as xgb, time

t0=time.time()
print("xgb", xgb.__version__)

def make_feats(view, s):
    tx = view.table("transactions")
    h = view.households
    if isinstance(h, pd.DataFrame):
        hh = pd.Index(h["household_key"].values, name="household_key")
    else:
        hh = pd.Index(list(h), name="household_key")
    f = pd.DataFrame(index=hh)
    eps = 1.0
    g = tx.groupby("household_key")
    f["tenure"] = (s - g["day"].min()).reindex(hh)
    f["days_since_last"] = (s - g["day"].max()).reindex(hh)
    f["spend_all"] = g["sales_value"].sum().reindex(hh)
    for w in (7,14,28,56,84,168):
        f[f"spend_{w}"] = tx.loc[tx["day"]>s-w].groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0.0)
    wk=[]
    for k in range(8):
        lo,hi = s-7*(k+1), s-7*k
        wk.append(tx.loc[(tx["day"]>lo)&(tx["day"]<=hi)].groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0.0))
    for k,a in enumerate(wk): f[f"wk{k}"]=a
    W=pd.concat(wk,axis=1)
    f["active_weeks8"]=(W>0).sum(axis=1).astype(float)
    f["wk_mean8"]=W.mean(axis=1); f["wk_std8"]=W.std(axis=1)
    f["wk_cv8"]=f["wk_std8"]/(f["wk_mean8"]+eps)
    for k,(lo,hi) in enumerate([(s-56,s-28),(s-84,s-56),(s-112,s-84)],start=1):
        f[f"spend28_lag{k}"]=tx.loc[(tx["day"]>lo)&(tx["day"]<=hi)].groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0.0)
    f["trend28"]=f["spend_28"]/(f["spend28_lag1"]+eps)
    f["trend84"]=f["spend_84"]/(f["spend_168"]/2+eps)
    f["share28_of_84"]=f["spend_28"]/(f["spend_84"]+eps)
    f["exp4w_blend"]=0.5*f["spend_28"]+0.3*f["spend28_lag1"]+0.2*f["spend28_lag2"]
    b84=tx.loc[tx["day"]>s-84].groupby(["household_key","basket_id"]).agg(sp=("sales_value","sum"), d=("day","max"))
    tb=b84.groupby("household_key")
    f["trips_84"]=tb.size().astype(float)
    f["basket_mean_84"]=tb["sp"].mean(); f["basket_max_84"]=tb["sp"].max(); f["basket_std_84"]=tb["sp"].std()
    ud=b84.reset_index().drop_duplicates(["household_key","d"])
    f["n_days_active_84"]=ud.groupby("household_key").size().reindex(hh).astype(float)
    gaps=ud.groupby("household_key")["d"].apply(lambda x: np.mean(np.diff(np.sort(x.values))) if len(x)>1 else np.nan)
    gapx=ud.groupby("household_key")["d"].apply(lambda x: np.max(np.diff(np.sort(x.values))) if len(x)>1 else np.nan)
    f["gap_mean_84"]=gaps.reindex(hh); f["gap_max_84"]=gapx.reindex(hh)
    b28=tx.loc[tx["day"]>s-28].groupby(["household_key","basket_id"])["sales_value"].sum()
    f["trips_28"]=b28.groupby("household_key").size().reindex(hh).fillna(0.0)
    f["spend_per_trip28"]=f["spend_28"]/(f["trips_28"]+eps)
    t84=tx.loc[tx["day"]>s-84]
    f["nprod_84"]=t84.groupby("household_key")["product_id"].nunique().reindex(hh).fillna(0.0)
    f["nstore_84"]=t84.groupby("household_key")["store_id"].nunique().reindex(hh).fillna(0.0)
    prods=view.table("products")[["product_id","department"]]
    m=t84[["household_key","product_id"]].merge(prods,on="product_id",how="left")
    f["ndept_84"]=m.groupby("household_key")["department"].nunique().reindex(hh).fillna(0.0)
    dsum=t84.groupby("household_key")[["coupon_disc","retail_disc","coupon_match_disc"]].sum().reindex(hh).fillna(0.0)
    for c in dsum.columns: f[c+"_84"]=dsum[c]
    f["disc_share_84"]=dsum.sum(axis=1)/(f["spend_84"]+eps)
    f["morn_share_84"]=t84.assign(morn=(t84["trans_time"]<1200).astype(float)).groupby("household_key")["morn"].mean().reindex(hh)
    f["weekly_rate_all"]=f["spend_all"]/(f["tenure"].clip(lower=1))
    f["exp4w_all"]=f["weekly_rate_all"]*4
    f["exp4w_168"]=f["spend_168"]/6.0
    f["exp4w_84"]=f["spend_84"]/3.0
    dem=view.table("demographics")
    if dem is not None and len(dem):
        d=dem.set_index("household_key")
        f["has_demo"]=f.index.isin(d.index).astype(float)
        for c in ["classification_1","classification_3","classification_4","classification_5"]:
            f[c]=pd.to_numeric(d[c].astype(str).str.extract(r"(\d+)")[0],errors="coerce").reindex(hh)
        f["classification_2"]=d["classification_2"].astype("category").cat.codes.reindex(hh).replace(-1,np.nan)
        ho={"Homeowner":4,"Probable Owner":3,"Probable Renter":2,"Renter":1,"Unknown":0}
        f["homeowner"]=d["homeowner_desc"].map(ho).reindex(hh)
        kid={"None/Unknown":0,"1":1,"2":2,"3+":3}
        f["kids"]=d["kid_category_desc"].map(kid).reindex(hh)
    else:
        f["has_demo"]=0.0
    f["snap_day"]=float(s); f["snap_week"]=float(view.week); f["snap_cycle_pos"]=float(s%28)
    return f

feats = agent_api.build_features(make_feats)
print("feats", feats.shape, "build secs", round(time.time()-t0,1))
pf = agent_api.save_table(feats.reset_index(), "feats_v3")
print("saved feats:", pf)

data = feats.reset_index()
tt = agent_api.train_targets()
data = data.merge(tt, on=["household_key","snapshot_day"], how="left")
feat_cols=[c for c in data.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
days = agent_api.snapshot_days()
tr = data[data.snapshot_day.isin(days["train"])]
va = data[data.snapshot_day.isin(days["validation"])]
print("train rows", len(tr), "val rows", len(va), "nan target", tr["future_spend_4w"].isna().sum())
X_tr, y_tr = tr[feat_cols], tr["future_spend_4w"].values
X_va = va[feat_cols]

major = int(xgb.__version__.split(".")[0])
obj = "reg:absoluteerror" if major>=2 else "reg:pseudohubererror"
base = dict(objective=obj, tree_method="hist", learning_rate=0.05, max_depth=6,
            min_child_weight=10, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.5,
            n_estimators=4000)
if obj=="reg:pseudohubererror": base["huber_slope"]=1.0
print("objective:", obj)

es_tr = tr[tr.snapshot_day<=403]; es_va = tr[tr.snapshot_day==431]
m1 = xgb.XGBRegressor(early_stopping_rounds=150, eval_metric="mae", **base)
m1.fit(es_tr[feat_cols], es_tr["future_spend_4w"].values, eval_set=[(es_va[feat_cols], es_va["future_spend_4w"].values)], verbose=False)
bi = m1.best_iteration
p_es = m1.predict(es_va[feat_cols])
print("best_iter", bi, "MAE@431(es)", round(np.abs(p_es-es_va["future_spend_4w"].values).mean(),3))

n_est = int(bi*1.15)+10
m2 = xgb.XGBRegressor(n_estimators=n_est, **base)
m2.fit(X_tr, y_tr, verbose=False)
pred = np.clip(m2.predict(X_va), 0, None)
print("pred stats: mean", round(pred.mean(),2), "median", round(np.median(pred),2), "max", round(pred.max(),1))

imp = pd.Series(m2.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(imp.head(20).round(4).to_string())

pp = agent_api.save_table(va[["household_key","snapshot_day"]].assign(prediction=pred), "pred_e004")
print("saved preds:", pp, "total secs", round(time.time()-t0,1))


# ---- cell ----

import numpy as np, pandas as pd, xgboost as xgb, time
t0=time.time()

feats = agent_api.load_saved("feats_v3.parquet")
data = feats
tt = agent_api.train_targets()
data = data.merge(tt, on=["household_key","snapshot_day"], how="left")
feat_cols=[c for c in data.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
days = agent_api.snapshot_days()
tr = data[data.snapshot_day.isin(days["train"])]
va = data[data.snapshot_day.isin(days["validation"])]
X_tr, y_tr = tr[feat_cols], tr["future_spend_4w"].values
X_va = va[feat_cols]

obj = "reg:absoluteerror"
base = dict(objective=obj, tree_method="hist", learning_rate=0.05, max_depth=6,
            min_child_weight=10, subsample=0.85, colsample_bytree=0.8, reg_lambda=1.5)

es_tr = tr[tr.snapshot_day<=403]; es_va = tr[tr.snapshot_day==431]
m1 = xgb.XGBRegressor(n_estimators=4000, early_stopping_rounds=150, eval_metric="mae", **base)
m1.fit(es_tr[feat_cols], es_tr["future_spend_4w"].values, eval_set=[(es_va[feat_cols], es_va["future_spend_4w"].values)], verbose=False)
bi = m1.best_iteration
p_es = m1.predict(es_va[feat_cols])
print("best_iter", bi, "MAE@431(es)", round(np.abs(p_es-es_va["future_spend_4w"].values).mean(),3))

n_est = int(bi*1.15)+10
m2 = xgb.XGBRegressor(n_estimators=n_est, **base)
m2.fit(X_tr, y_tr, verbose=False)
pred = np.clip(m2.predict(X_va), 0, None)
print("pred stats: mean", round(pred.mean(),2), "median", round(np.median(pred),2), "max", round(pred.max(),1))

imp = pd.Series(m2.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(imp.head(20).round(4).to_string())

pp = agent_api.save_table(va[["household_key","snapshot_day"]].assign(prediction=pred), "pred_e004")
print("saved preds:", pp, "total secs", round(time.time()-t0,1))
