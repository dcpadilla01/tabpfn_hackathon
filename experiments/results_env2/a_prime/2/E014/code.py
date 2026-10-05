import agent_api, pandas as pd, numpy as np

print("=== API ===")
print([x for x in dir(agent_api) if not x.startswith("_")])

print("\n=== E011 table ===")
try:
    t = agent_api.load_saved("e011_discounts.parquet")
    print("shape:", t.shape)
    print("cols:", list(t.columns))
except Exception as e:
    print("ERR", repr(e))

print("\n=== targets by snapshot ===")
tt = agent_api.train_targets()
print(tt.groupby("snapshot_day")["future_spend_4w"].agg(["count","mean","median","std","max"]))
print("zero share overall:", round((tt.future_spend_4w==0).mean(),4))

print("\n=== view format + departments ===")
v = agent_api.snapshot(459)
print(type(v.households), len(v.households), list(v.households[:3]))
tr_full = v.transactions
print("tr shape:", tr_full.shape)
p = v.table("products")
dept_spend = tr_full.merge(p[["product_id","department"]], on="product_id", how="left").groupby("department")["sales_value"].sum().sort_values(ascending=False)
print("top dept spend:", {k: round(x) for k,x in dept_spend.head(12).items()})


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
print("view type:", type(v))
print([x for x in dir(v) if not x.startswith("_")])
print("day:", v.day, "week:", v.week)
hh = v.households
print("households type:", type(hh))
try:
    import itertools
    print("first few:", list(itertools.islice(hh, 5)))
except Exception as e:
    print("iter err", repr(e))
tr = v.transactions
print("tr type:", type(tr), getattr(tr, "shape", None))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
p = v.table("products")
dept_spend = tr.merge(p[["product_id","department"]], on="product_id", how="left").groupby("department")["sales_value"].sum().sort_values(ascending=False)
print("dept spend (top 12):")
print(dept_spend.head(12).round(0))
print("\nn depts:", tr.merge(p[["product_id","department"]], on="product_id", how="left")["department"].nunique())

# top commodities
com = tr.merge(p[["product_id","commodity_desc"]], on="product_id", how="left").groupby("commodity_desc")["sales_value"].sum().sort_values(ascending=False)
print("\ntop commodities:")
print(com.head(15).round(0))

# store-level: n stores per household over 364d
hh_stores = tr[tr.day >= 459-364].groupby("household_key")["store_id"].nunique()
print("\nn_stores_364 distribution:", hh_stores.describe().round(2).to_dict())

# trans_time
print("\ntrans_time sample:", tr.trans_time.dropna().head().tolist())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
p = v.table("products")

# department spend shares over 364d, correlation with target
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w
print("n tgt:", len(tgt))

w = tr[tr.day >= 459-364]
w = w.merge(p[["product_id","department"]], on="product_id", how="left")
gs = w.groupby(["household_key","department"])["sales_value"].sum().unstack().fillna(0)
tot = gs.sum(axis=1)
shares = gs.div(tot, axis=0)

# target correlations
rows=[]
for c in shares.columns:
    common = tgt.index.intersection(shares.index)
    r = np.corrcoef(shares.loc[common, c], tgt.loc[common])[0,1]
    rows.append((c, round(r,3), round(shares[c].mean(),3)))
rows.sort(key=lambda x: -abs(x[1]))
print("dept share corr with target (364d window):")
for r in rows[:15]: print(r)

# store loyalty: main-store share of trips
st = tr[tr.day >= 459-364].groupby(["household_key","store_id"]).size()
main_share = st.groupby(level=0).max()/st.groupby(level=0).sum()
common = tgt.index.intersection(main_share.index)
print("\ncorr main-store trip share vs target:", round(np.corrcoef(main_share.loc[common], tgt.loc[common])[0,1],3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w
common = tgt.index.intersection(tr.household_key.unique())

# KIOSK-GAS spend share and absolute
w = tr[tr.day >= 459-364].merge(v.table("products")[["product_id","department"]], on="product_id", how="left")
gs = w.groupby(["household_key","department"])["sales_value"].sum().unstack().fillna(0)
gas_abs = gs.get("KIOSK-GAS", pd.Series(0, index=gs.index))
gas_share = gas_abs/gs.sum(axis=1)
print("corr gas_abs:", round(np.corrcoef(gas_abs.loc[common], tgt.loc[common])[0,1],3))
print("corr gas_share:", round(np.corrcoef(gas_share.loc[common], tgt.loc[common])[0,1],3))

# spend in trailing 7d, 14d — already in E011? E011 has spend_7, spend_14. Check corr of E011 features with target at 431
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
t431 = tgt.reindex(E431.index)
corrs = E431.drop(columns=["household_key","snapshot_day"]).corrwith(t431).sort_values()
print("\nlowest corr E011 feats:")
print(corrs.head(8).round(3))
print("highest corr E011 feats:")
print(corrs.tail(12).round(3))

# how much signal in spend_28 alone? residual analysis
import numpy.linalg as la
X = E431[["spend_28"]].fillna(0).values
y = t431.values
b = la.lstsq(X, y, rcond=None)[0]
res = y - X@b
print("\nMAE of spend_28*coef at 431:", round(np.abs(res).mean(),2), "coef:", round(b[0],3))
print("MAE of median:", round(np.abs(y-np.median(y)).mean(),2))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w

w = tr[tr.day >= 459-364].merge(v.table("products")[["product_id","department"]], on="product_id", how="left")
gs = w.groupby(["household_key","department"])["sales_value"].sum().unstack().fillna(0)
gas_abs = gs["KIOSK-GAS"] if "KIOSK-GAS" in gs.columns else pd.Series(0, index=gs.index)
gas_share = gas_abs/gs.sum(axis=1)
common = tgt.index.intersection(gs.index)
print("corr gas_abs:", round(np.corrcoef(gas_abs.loc[common], tgt.loc[common])[0,1],3))
print("corr gas_share:", round(np.corrcoef(gas_share.loc[common], tgt.loc[common])[0,1],3))

E = agent_api.load_saved("e011_discounts.parquet")
print("E011 snap days:", sorted(E.snapshot_day.unique()))
E431 = E[E.snapshot_day==431].set_index("household_key")
t431 = tgt.reindex(E431.index)
feats = [c for c in E431.columns if c not in ("household_key","snapshot_day")]
corrs = E431[feats].apply(lambda s: pd.to_numeric(s, errors="coerce")).corrwith(t431).sort_values()
print("\nlowest corr E011 feats:"); print(corrs.head(8).round(3))
print("highest corr E011 feats:"); print(corrs.tail(12).round(3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
p = v.table("products")

print("sales_value stats:", tr.sales_value.describe().round(2).to_dict())
print("neg sales rows:", (tr.sales_value<0).sum(), " zero:", (tr.sales_value==0).sum())
print("quantity stats:", tr.quantity.describe().round(2).to_dict())
print("qty<=0 rows:", (tr.quantity<=0).sum())

# trans_time -> hour
tt = tr.trans_time.dropna()
hh_ = (tt//100).clip(0,23)
print("\nhour dist:", hh_.value_counts().sort_index().to_dict())

# day-of-week (day mod 7) trip counts
dow = tr.groupby(tr.day % 7)["basket_id"].nunique()
print("\nbaskets by day%7:", dow.to_dict())

# brand
print("\nbrand values:", p.brand.value_counts(dropna=False).to_dict())

# COUPON/MISC ITEMS commodity check
cm = tr.merge(p[["product_id","commodity_desc"]], on="product_id", how="left")
print("\nCOUPON/MISC stats:", cm[cm.commodity_desc=="COUPON/MISC ITEMS"].sales_value.describe().round(2).to_dict())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w
common = tgt.index.intersection(tr.household_key.unique())

# basket-level aggregation: max basket, top-2 basket share, basket count
b = tr[tr.day >= 459-364].groupby(["household_key","basket_id"]).agg(s=("sales_value","sum"), d=("day","first"))
per_hh = b.groupby("household_key")["s"]
max_b = per_hh.max(); top2 = b.sort_values("s", ascending=False).groupby("household_key").head(2).groupby("household_key")["s"].sum()
tot = per_hh.sum()
common2 = tgt.index.intersection(tot.index)
print("corr max_basket:", round(np.corrcoef(max_b.loc[common2], tgt.loc[common2])[0,1],3))
print("corr top2_baskets:", round(np.corrcoef(top2.loc[common2], tgt.loc[common2])[0,1],3))
print("corr top2_share:", round(np.corrcoef((top2/tot).loc[common2], tgt.loc[common2])[0,1],3))

# trans_time: weekend share of spend (day%7 in {4,5})
w = tr[tr.day >= 459-364].copy()
w["wknd"] = (w.day % 7).isin([4,5]).astype(int)
wknd_spend = w.groupby("household_key").apply(lambda g: (g.sales_value*g.wknd).sum()/g.sales_value.sum(), include_groups=False)
print("corr weekend spend share:", round(np.corrcoef(wknd_spend.loc[common2], tgt.loc[common2])[0,1],3))

# hour: evening share (trans_time>=1700)
w["eve"] = (w.trans_time>=1700).astype(int)
eve_spend = w.groupby("household_key").apply(lambda g: (g.sales_value*g.eve).sum()/g.sales_value.sum(), include_groups=False)
print("corr evening spend share:", round(np.corrcoef(eve_spend.loc[common2], tgt.loc[common2])[0,1],3))

# gas absolute spend corr at 28d window too
gas28 = tr[tr.day >= 459-28].merge(v.table("products")[["product_id","department"]], on="product_id", how="left")
g28 = gas28[gas28.department=="KIOSK-GAS"].groupby("household_key")["sales_value"].sum()
g28 = g28.reindex(tgt.index).fillna(0)
print("corr gas_spend_28 vs target:", round(np.corrcoef(g28, tgt)[0,1],3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
p = v.table("products")
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w

# residual after best linear combo of E011's top spend features at 431
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
t431 = tgt.reindex(E431.index)
base_feats = ["spend_28","spend_84","spend_180","ew_28","ew_84","ew_180","spend_365","lr_mean28","avg_basket_84","baskets_28","baskets_84","trips_per_wk_84","spend_364","rd_84","rd_364"]
X = E431[base_feats].apply(pd.to_numeric, errors="coerce").fillna(0).values
X = np.column_stack([np.ones(len(X)), X])
y = t431.values
bcoef, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X@bcoef
print("base MAE at 431:", round(np.abs(res).mean(),3))

# candidate new features on 364d window, corr with residual
w = tr[tr.day >= 459-364].merge(p[["product_id","department"]], on="product_id", how="left")
gs = w.groupby(["household_key","department"])["sales_value"].sum().unstack().fillna(0)
cands = {}
cands["gas_abs_364"] = gs["KIOSK-GAS"]
cands["gas_share_364"] = gs["KIOSK-GAS"]/gs.sum(axis=1)
b = w.groupby(["household_key","basket_id"]).agg(s=("sales_value","sum"))
cands["max_basket_364"] = b.groupby("household_key")["s"].max()
cands["top2_baskets_364"] = b.sort_values("s", ascending=False).groupby("household_key").head(2).groupby("household_key")["s"].sum()
cands["n_lines_364"] = w.groupby("household_key").size()
cands["n_distinct_products_364"] = w.groupby("household_key")["product_id"].nunique()
cands["units_364"] = w.groupby("household_key")["quantity"].sum()
cands["std_basket_364"] = b.groupby("household_key")["s"].std()
cands["iqr_basket_364"] = b.groupby("household_key")["s"].quantile(0.75)-b.groupby("household_key")["s"].quantile(0.25)
cands["p90_basket_364"] = b.groupby("household_key")["s"].quantile(0.9)

for k, s in cands.items():
    s = s.reindex(E431.index).fillna(0) if k!="std_basket_364" else s.reindex(E431.index)
    cc = np.corrcoef(pd.to_numeric(s, errors="coerce").fillna(0), res)[0,1]
    print(f"{k}: corr_res={cc:.3f}")


# ---- cell ----
import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w

# Correlation of TARGET with its own lag-1 (spend in 28d window ending at snapshot) vs other windows
# Also: how noisy is the target? autocorr of household 28d spend series
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
t431 = tgt.reindex(E431.index)

# ratio target / spend_28: distribution
r = (t431/(E431.spend_28+1))
print("target/spend_28 ratio quantiles:", r.quantile([0.1,0.25,0.5,0.75,0.9]).round(2).to_dict())

# distribution of target by decile of spend_28: how much does model need to shrink?
qs = pd.qcut(E431.spend_28, 10, duplicates="drop")
print("\nmean target by spend_28 decile:")
print(t431.groupby(qs).agg(["mean","count"]).round(1))

# what fraction of target variance explained by spend_28 alone (R2)
X = E431.spend_28.values
r2 = np.corrcoef(X, t431)[0,1]**2
print("\nR2 spend_28 vs target:", round(r2,3))
# and best linear combo of all numeric E011
Xall = E431.drop(columns=["household_key","snapshot_day"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
Xall = np.column_stack([np.ones(len(Xall)), Xall])
b, *_ = np.linalg.lstsq(Xall, t431.values, rcond=None)
pred = Xall@b
print("in-sample R2 all E011 feats at 431:", round(1 - ((t431-pred)**2).sum()/((t431-t431.mean())**2).sum(),3))
print("in-sample MAE:", round(np.abs(t431-pred).mean(),2))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
tt = agent_api.train_targets()
t431 = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w.reindex(E431.index)

Xall = E431.drop(columns=["snapshot_day"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
Xall = np.column_stack([np.ones(len(Xall)), Xall])
y = t431.values
b, *_ = np.linalg.lstsq(Xall, y, rcond=None)
pred = Xall@b
print("in-sample R2 all E011 feats at 431:", round(1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum(),3))
print("in-sample MAE:", round(np.abs(y-pred).mean(),2))

# per-snapshot OOS-ish check: fit on train snapshots !=431, eval at 431
tr_days = [d for d in sorted(E.snapshot_day.unique()) if d<431]
trX = E[E.snapshot_day.isin(tr_days)]
Xtr = trX.drop(columns=["snapshot_day"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
ytr = tt.set_index(["snapshot_day","household_key"]).loc[[(d,h) for d,h in zip(trX.snapshot_day, trX.household_key)]].values
b2, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr)), Xtr]), ytr, rcond=None)
Xt = np.column_stack([np.ones(len(Xall)), Xall])
pred2 = Xt@b2
print("holdout-431 MAE (linear, all E011):", round(np.abs(y-pred2).mean(),2))

# same but only spend features
sp = ["spend_28","spend_84","spend_180","spend_365","ew_28","ew_84","ew_180","spend_28_prior","spend_84_prior","spend_lag364","lr_mean28","avg_basket_84","baskets_28","baskets_84","trips_per_wk_84","spend_364","rd_84","rd_364","spend_7","spend_14"]
Xs = E431[sp].apply(pd.to_numeric, errors="coerce").fillna(0).values
Xtr_s = trX[sp].apply(pd.to_numeric, errors="coerce").fillna(0).values
b3, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xtr_s)), Xtr_s]), ytr, rcond=None)
pred3 = np.column_stack([np.ones(len(Xs)), Xs])@b3
print("holdout-431 MAE (linear, spend-only subset):", round(np.abs(y-pred3).mean(),2))


# ---- cell ----
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


# ---- cell ----
import agent_api, pandas as pd, numpy as np
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
tt = agent_api.train_targets()
tmap = tt.set_index(["snapshot_day","household_key"]).future_spend_4w

def num(df):
    return df.drop(columns=["snapshot_day"]).apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

y = t431.values
tr_days = [d for d in sorted(E.snapshot_day.unique()) if d<431]
trX = E[E.snapshot_day.isin(tr_days)]
Xtr = num(trX).values
ytr = tmap.loc[list(zip(trX.snapshot_day.astype(int), trX.household_key))].values

def fit_eval(cols_tr, cols_t, name):
    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(cols_tr)), cols_tr]), ytr, rcond=None)
    pred = np.column_stack([np.ones(len(cols_t)), cols_t])@b
    print(name, "holdout-431 MAE:", round(np.abs(y-pred).mean(),2))
    return pred

Xall_t = num(E431).values
fit_eval(Xtr, Xall_t, "all-E011")
sp = ["spend_28","spend_84","spend_180","spend_365","ew_28","ew_84","ew_180","spend_28_prior","spend_84_prior","spend_lag364","lr_mean28","avg_basket_84","baskets_28","baskets_84","trips_per_wk_84","spend_364","rd_84","rd_364","spend_7","spend_14"]
fit_eval(trX[sp].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values,
         E431[sp].apply(pd.to_numeric, errors="coerce").fillna(0).astype(float).values, "spend-only")

# also: what's the val MAE if we just predict spend_28*shrinkage? proxy
print("\nval rows in E011:", (E.snapshot_day>=459).sum())


# ---- cell ----
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


# ---- cell ----
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
