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
