import agent_api as api, pandas as pd, numpy as np

v = api.snapshot()
c = v.table("campaigns")
print("campaigns", c.shape)
print(c.head(3).to_string())
print(c["description"].value_counts())
print(c[["start_day","end_day"]].describe().to_string())
ct = v.table("campaign_targets")
print("targets", ct.shape)
print(ct["description"].value_counts())
tx = v.table("transactions")
print(tx[["sales_value","retail_disc","coupon_disc","coupon_match_disc"]].describe().to_string())

base = api.load_saved("e006_newblock.parquet")
print("base", base.shape, base.columns[:6].tolist())


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
for name in ["e006_zero_inflation.parquet","e006_newblock.parquet","e007_ar_lags.parquet","e008_log_transform.parquet"]:
    t = api.load_saved(name)
    print(name, t.shape, "dup:", t.duplicated(["household_key","snapshot_day"]).sum())
b = api.load_saved("e006_zero_inflation.parquet")
print(b.columns.tolist())


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    s = snapshot_day
    hh = pd.Index(view.households, name="household_key")
    c = view.table("campaigns")            # start_day <= s
    ct = view.table("campaign_targets")
    cm = ct.merge(c[["campaign","description","start_day","end_day"]], on="campaign", how="left")
    cm = cm[cm.household_key.isin(hh)]
    g = cm.groupby("household_key")
    feat = pd.DataFrame(index=hh)
    feat["c_active_n"] = g.apply(lambda d: ((d.end_day >= s)).sum()).reindex(hh).fillna(0)
    for t in ["TypeA","TypeB","TypeC"]:
        feat[f"c_act_{t}"] = g.apply(lambda d: int(((d.description==t) & (d.end_day>=s)).any())).reindex(hh).fillna(0)
    feat["c_days_since_start"] = (s - g.start_day.max()).reindex(hh)      # most recent campaign start
    nxt = cm[cm.end_day >= s].groupby("household_key").end_day.min()
    feat["c_days_to_end"] = (nxt - s).reindex(hh)
    feat["c_started_28"] = g.apply(lambda d: int((d.start_day >= s-27).any())).reindex(hh).fillna(0)
    feat["c_started_84"] = g.apply(lambda d: int((d.start_day >= s-83).any())).reindex(hh).fillna(0)
    # recent spend for interactions
    tx = view.table("transactions")
    r = tx[(tx.day > s-28) & (tx.day <= s)].groupby("household_key").sales_value.sum()
    feat["spend28_c"] = r.reindex(hh).fillna(0.0)
    feat["c_actA_x_spend28"] = feat["c_act_TypeA"] * feat["spend28_c"]
    feat["c_actB_x_spend28"] = feat["c_act_TypeB"] * feat["spend28_c"]
    feat["c_actA_x_dec"] = feat["c_act_TypeA"] * feat["c_days_since_start"].fillna(999)
    feat["c_anyn_x_spend28"] = (feat["c_active_n"] > 0).astype(int) * feat["spend28_c"]
    return feat.drop(columns=["spend28_c"])

newf = api.build_features(fn)
print(newf.shape, newf.snapshot_day.unique())
print(newf.groupby("snapshot_day")[["c_active_n","c_act_TypeA","c_act_TypeB","c_days_to_end","c_started_28"]].mean())


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    s = snapshot_day
    hh = pd.Index(view.households, name="household_key")
    c = view.table("campaigns")
    ct = view.table("campaign_targets")
    cm = ct.merge(c[["campaign","description","start_day","end_day"]], on="campaign", how="left")
    cm["description"] = cm["description"].astype(str)
    cm = cm[cm.household_key.isin(hh)]
    g = cm.groupby("household_key")
    feat = pd.DataFrame(index=hh)
    feat["c_active_n"] = g.apply(lambda d: float((d.end_day >= s).sum())).reindex(hh).fillna(0.0)
    for t in ["TypeA","TypeB","TypeC"]:
        sub = cm[cm.description == t]
        feat[f"c_act_{t}"] = sub.groupby("household_key").end_day.max().ge(s).astype(float).reindex(hh).fillna(0.0)
    feat["c_days_since_start"] = (s - g.start_day.max()).reindex(hh).astype(float)
    nxt = cm[cm.end_day >= s].groupby("household_key").end_day.min()
    feat["c_days_to_end"] = (nxt - s).reindex(hh).astype(float)
    feat["c_started_28"] = g.apply(lambda d: float((d.start_day >= s-27).any())).reindex(hh).fillna(0.0)
    feat["c_started_84"] = g.apply(lambda d: float((d.start_day >= s-83).any())).reindex(hh).fillna(0.0)
    tx = view.table("transactions")
    r = tx[(tx.day > s-28) & (tx.day <= s)].groupby("household_key").sales_value.sum()
    sp = r.reindex(hh).fillna(0.0)
    feat["c_actA_x_spend28"] = feat["c_act_TypeA"] * sp
    feat["c_actB_x_spend28"] = feat["c_act_TypeB"] * sp
    feat["c_actA_x_dec"] = feat["c_act_TypeA"] * feat["c_days_since_start"].fillna(999.0)
    feat["c_anyn_x_spend28"] = (feat["c_active_n"] > 0).astype(float) * sp
    return feat

newf = api.build_features(fn)
print(newf.shape, sorted(newf.snapshot_day.unique()))
print(newf.groupby("snapshot_day")[["c_active_n","c_act_TypeA","c_act_TypeB","c_days_to_end","c_started_28"]].mean().round(2).to_string())
path = api.save_table(newf, "e009_campaign_dynamics")
print(path)


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    s = snapshot_day
    hh = pd.Index(view.households, name="household_key")
    c = view.table("campaigns")[["campaign","start_day","end_day"]]
    ct = view.table("campaign_targets")
    cm = ct.merge(c, on="campaign", how="left")
    cm = cm[cm.household_key.isin(hh)]
    g = cm.groupby("household_key")
    feat = pd.DataFrame(index=hh)
    feat["c_active_n"] = g.apply(lambda d: float((d.end_day >= s).sum())).reindex(hh).fillna(0.0)
    for t in ["TypeA","TypeB","TypeC"]:
        sub = cm[cm.description == t]
        feat[f"c_act_{t}"] = sub.groupby("household_key").end_day.max().ge(s).astype(float).reindex(hh).fillna(0.0)
    feat["c_days_since_start"] = (s - g.start_day.max()).reindex(hh).astype(float)
    nxt = cm[cm.end_day >= s].groupby("household_key").end_day.min()
    feat["c_days_to_end"] = (nxt - s).reindex(hh).astype(float)
    feat["c_started_28"] = g.apply(lambda d: float((d.start_day >= s-27).any())).reindex(hh).fillna(0.0)
    feat["c_started_84"] = g.apply(lambda d: float((d.start_day >= s-83).any())).reindex(hh).fillna(0.0)
    tx = view.table("transactions")
    r = tx[(tx.day > s-28) & (tx.day <= s)].groupby("household_key").sales_value.sum()
    sp = r.reindex(hh).fillna(0.0)
    feat["c_actA_x_spend28"] = feat["c_act_TypeA"] * sp
    feat["c_actB_x_spend28"] = feat["c_act_TypeB"] * sp
    feat["c_actA_x_dec"] = feat["c_act_TypeA"] * feat["c_days_since_start"].fillna(999.0)
    feat["c_anyn_x_spend28"] = (feat["c_active_n"] > 0).astype(float) * sp
    return feat

newf = api.build_features(fn)
print(newf.shape, sorted(newf.snapshot_day.unique()))
print(newf.groupby("snapshot_day")[["c_active_n","c_act_TypeA","c_act_TypeB","c_days_to_end","c_started_28"]].mean().round(2).to_string())
path = api.save_table(newf, "e009_campaign_dynamics")
print(path)


# ---- cell ----
import agent_api as api
v = api.snapshot()
for name in ["campaigns","campaign_targets","transactions"]:
    t = v.table(name)
    print(name, dict(t.dtypes.astype(str)))


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
v = api.snapshot(95)
hh = pd.Index(v.households)
print(type(v.households), type(hh), hh.dtype)
c = v.table("campaigns")[["campaign","start_day","end_day"]]
ct = v.table("campaign_targets")
cm = ct.merge(c, on="campaign", how="left")
cm = cm[cm.household_key.isin(hh)]
g = cm.groupby("household_key")
r = g.apply(lambda d: float((d.end_day >= 95).sum()))
print("apply result dtype:", r.dtype, type(r.index), r.index.dtype)
print(type(g.keys))


# ---- cell ----
import agent_api as api
v = api.snapshot()
print(type(v.households), getattr(v.households, "dtype", None), len(v.households) if v.households is not None else None)
print(v.day, v.week)
hh = pd.Index(v.households)
print(hh.dtype)
c = v.table("campaigns")[["campaign","start_day","end_day"]]
ct = v.table("campaign_targets")
cm = ct.merge(c, on="campaign", how="left")
cm = cm[cm.household_key.isin(hh)]
g = cm.groupby("household_key")
r = g.apply(lambda d: float((d.end_day >= 459).sum()))
print("dtype:", r.dtype, "idx dtype:", r.index.dtype, "len:", len(r))
print(r.head())


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np

def probe(view, s):
    hh = view.households
    print("households type:", type(hh), getattr(hh, "dtype", None))
    c = view.table("campaigns")[["campaign","start_day","end_day"]]
    ct = view.table("campaign_targets")
    cm = ct.merge(c, on="campaign", how="left")
    print("cm household dtype:", cm.household_key.dtype)
    cm = cm[cm.household_key.isin(hh)]
    g = cm.groupby("household_key")
    r = g.apply(lambda d: float((d.end_day >= s).sum()))
    print("r dtype:", r.dtype, "index dtype:", r.index.dtype, "is_cat_idx:", isinstance(r.index, pd.CategoricalIndex))
    r2 = r.reindex(hh)
    print("r2 dtype:", r2.dtype, "idx:", type(r2.index))
    print("hh is index?", isinstance(hh, pd.Index), "categorical?", isinstance(hh, pd.CategoricalIndex))

api.build_features(probe)


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np

def fn(view, s):
    hh = view.households
    c = view.table("campaigns")[["campaign","start_day","end_day"]]
    ct = view.table("campaign_targets")
    cm = ct.merge(c, on="campaign", how="left")
    cm = cm[cm.household_key.isin(hh)]
    feat = pd.DataFrame(index=hh)
    act = cm[cm.end_day >= s]
    feat["c_active_n"] = act.groupby("household_key").size().reindex(hh).fillna(0.0)
    for t in ["TypeA","TypeB","TypeC"]:
        m = (cm.description == t).values
        sub = cm[m]
        feat[f"c_act_{t}"] = sub.groupby("household_key").end_day.max().ge(s).astype(float).reindex(hh).fillna(0.0)
    feat["c_days_since_start"] = (s - cm.groupby("household_key").start_day.max()).reindex(hh).astype(float)
    feat["c_days_to_end"] = (act.groupby("household_key").end_day.min() - s).reindex(hh).astype(float)
    feat["c_started_28"] = cm[cm.start_day >= s-27].groupby("household_key").size().gt(0).astype(float).reindex(hh).fillna(0.0)
    feat["c_started_84"] = cm[cm.start_day >= s-83].groupby("household_key").size().gt(0).astype(float).reindex(hh).fillna(0.0)
    tx = view.table("transactions")
    sp = tx[(tx.day > s-28) & (tx.day <= s)].groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)
    feat["c_actA_x_spend28"] = feat["c_act_TypeA"] * sp
    feat["c_actB_x_spend28"] = feat["c_act_TypeB"] * sp
    feat["c_actA_x_dec"] = feat["c_act_TypeA"] * feat["c_days_since_start"].fillna(999.0)
    feat["c_anyn_x_spend28"] = (feat["c_active_n"] > 0).astype(float) * sp
    return feat

newf = api.build_features(fn)
print(newf.shape, sorted(newf.snapshot_day.unique()))
print(newf.groupby("snapshot_day")[["c_active_n","c_act_TypeA","c_act_TypeB","c_days_to_end","c_started_28"]].mean().round(2).to_string())
path = api.save_table(newf, "e009_campaign_dynamics")
print(path)


# ---- cell ----
import agent_api as api, pandas as pd, numpy as np
base = api.load_saved("e006_newblock.parquet")
newf = api.load_saved("e009_campaign_dynamics.parquet")
m = base.merge(newf, on=["household_key","snapshot_day"], how="left")
print(m.shape, "dup:", m.duplicated(["household_key","snapshot_day"]).sum())
print(m[["c_active_n","c_act_TypeA","c_days_to_end","c_started_28"]].isna().mean().round(3).to_dict())
path = api.save_table(m, "e009_full")
print(path)
