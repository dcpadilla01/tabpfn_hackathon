import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
BASE = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
        objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")
td = [d for d in range(95,432,28) if d < 431]
m = xgb.XGBRegressor(**BASE).fit(df[df.snapshot_day.isin(td)][FEATS], df[df.snapshot_day.isin(td)].future_spend_4w)
pr = df[df.snapshot_day==431].copy()
pr["pred"] = m.predict(pr[FEATS])
pr["resid"] = pr.future_spend_4w - pr.pred

# candidate features from raw transactions at snapshot 431
v = A.snapshot(431)
tx = v.table("transactions")
hhs = pr.household_key.values
tx = tx[tx.household_key.isin(hhs)]

def build(day):
    t = tx[tx.day <= day]
    g = t.groupby("household_key")
    out = pd.DataFrame(index=g.size().index)
    # last trip spend & day
    bs = t.groupby(["household_key","basket_id"]).agg(sp=("sales_value","sum"), d=("day","max"))
    last = bs.groupby("household_key").agg(last_sp=("sp","last"), last_d=("d","max"))
    out["last_trip_sp"] = last.last_sp
    out["days_last_trip"] = day - last.last_d
    for w in [7,14]:
        s = t[t.day > day-w].groupby("household_key").sales_value.sum().rename(f"s{w}")
        out = out.join(s)
    out["accel"] = out.s7/(out.s14/2+1e-9)
    # max basket in 28d
    b28 = bs[bs.d > day-28]
    out["max_basket28"] = b28.groupby("household_key").sp.max()
    out["n_trips14"] = b28[b28.d > day-14].groupby("household_key").size()
    # gap regularity: std of trip days in 84d
    b84 = bs[bs.d > day-84].reset_index()
    out["gap_std_84"] = b84.groupby("household_key").d.apply(lambda s: s.sort_values().diff().std())
    # dept HHI 84d
    t84 = t[t.day > day-84].merge(v.table("products")[["product_id","department"]], on="product_id", how="left")
    ds = t84.groupby(["household_key","department"]).sales_value.sum().reset_index()
    tot = ds.groupby("household_key").sales_value.transform("sum")
    ds["sh"] = (ds.sales_value/tot)**2
    out["dept_hhi"] = ds.groupby("household_key").sh.sum()
    # private label share 84d
    t84b = t84.merge(v.table("products")[["product_id","brand"]], on="product_id", how="left", suffixes=("","_b"))
    sh_pl = t84b.assign(pl=(t84b.brand=="Private").astype(float)).groupby("household_key").apply(
        lambda g: (g.sales_value*g.pl).sum()/g.sales_value.sum())
    out["pl_share"] = sh_pl
    # quantity per trip 28d
    q = t[t.day > day-28].groupby("household_key").quantity.sum()
    out["qty28"] = q
    return out

cf = build(431)
pr2 = pr.set_index("household_key").join(cf)
sub = pr2[pr2.days_since_last<=28]
print("corr(resid, cand) | corr(cand, target) among dsl<=28 (n=%d):" % len(sub))
for c in cf.columns:
    r1 = np.corrcoef(sub[c].fillna(sub[c].median()), sub.resid)[0,1]
    r2 = np.corrcoef(sub[c].fillna(sub[c].median()), sub.future_spend_4w)[0,1]
    print(f"  {c:14s} resid {r1:+.3f}   target {r2:+.3f}")
