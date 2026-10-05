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
