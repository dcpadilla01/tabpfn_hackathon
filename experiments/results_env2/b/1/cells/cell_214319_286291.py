import pandas as pd, numpy as np, re, agent_api

snap = agent_api.snapshot()
ct = snap.table("campaign_targets")
print("campaign_targets", ct.shape); print(ct.head(3))
print(ct.description.value_counts())
camps = snap.table("campaigns")
print("campaigns", camps.shape); print(camps.head(3))
red = snap.table("coupon_redemptions")
print("redemptions", red.shape); print(red.head(3))

def mk(view, snapshot_day):
    hh = pd.Index(view.households, name="household_key")
    trans = view.table("transactions")
    camps = view.table("campaigns")
    ct = view.table("campaign_targets")
    red = view.table("coupon_redemptions")

    active_set = set(camps.loc[(camps.start_day <= snapshot_day) & (camps.end_day >= snapshot_day), "campaign"])
    recent_set = set(camps.loc[camps.start_day > snapshot_day - 84, "campaign"])

    ct2 = ct[ct.household_key.isin(hh)].copy()
    ct2["is_active"] = ct2.campaign.isin(active_set)
    ct2["is_recent"] = ct2.campaign.isin(recent_set)
    agg = ct2.groupby("household_key").agg(
        n_camp_targeted=("campaign", "nunique"),
        n_camp_active=("is_active", "sum"),
        n_camp_recent=("is_recent", "sum"),
    )
    type_ct = pd.crosstab(ct2.household_key, ct2.description)
    type_ct.columns = ["ct_type_" + re.sub(r"\W+", "_", str(c)) for c in type_ct.columns]

    r = red[red.household_key.isin(hh) & (red.day <= snapshot_day)]
    rdf = pd.DataFrame(index=hh)
    rdf["n_red_total"] = r.groupby("household_key").size()
    for w in (28, 56, 84, 168):
        rdf[f"n_red_{w}"] = r[r.day > snapshot_day - w].groupby("household_key").size()
    rdf["days_since_red"] = snapshot_day - r.groupby("household_key").day.max()
    rdf["n_red_camps"] = r.groupby("household_key").campaign.nunique()

    t = trans[trans.household_key.isin(hh)]
    out = pd.DataFrame(index=hh)
    for name, lo in (("28", snapshot_day-28), ("56", snapshot_day-56), ("84", snapshot_day-84)):
        tw = t[t.day > lo]
        g = tw.groupby("household_key").agg(sp=("sales_value", "sum"), cd=("coupon_disc", "sum"),
                                            cmd=("coupon_match_disc", "sum"), rd=("retail_disc", "sum"))
        sp = g.sp.replace(0, np.nan)
        out[f"coupon_disc_share_{name}"] = -(g.cd) / sp
        out[f"retail_disc_share_{name}"] = -(g.rd) / sp
        out[f"deal_disc_share_{name}"] = -(g.cd + g.cmd + g.rd) / sp
    out = out.join(rdf).join(agg).join(type_ct)
    return out

base = agent_api.load_saved("e003_full.parquet")
print("base", base.shape)
new = agent_api.build_features(mk)
print("new", new.shape)
m = base.merge(new.reset_index(), on=["household_key", "snapshot_day"], how="left")
print("merged", m.shape)
path = agent_api.save_table(m, "e004_marketing")
print(path)
