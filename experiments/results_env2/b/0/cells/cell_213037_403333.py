import agent_api as A
import pandas as pd, numpy as np

def fn(view, snapshot_day):
    hh = view.households
    idx = pd.Index(hh, name="household_key")
    out = pd.DataFrame(index=idx)

    # campaigns already started that targeted these households
    camps = view.table("campaigns")            # start_day <= snapshot_day
    tgt = view.table("campaign_targets")
    tgt = tgt[tgt.household_key.isin(hh)]
    t = tgt.merge(camps[["campaign","description","start_day"]], on="campaign", how="left")
    t = t[t.start_day <= snapshot_day]
    g = t.groupby("household_key")
    out["n_campaigns"] = g.size().reindex(idx).fillna(0)
    for d in ["TypeA","TypeB","TypeC"]:
        out["n_camp_"+d] = t[t.description==d].groupby("household_key").size().reindex(idx).fillna(0)
    rec = t[t.start_day > snapshot_day-56]
    out["camp_recent"] = rec.groupby("household_key").size().reindex(idx).fillna(0)

    # coupon redemptions
    cr = view.table("coupon_redemptions")
    cr = cr[cr.household_key.isin(hh)]
    out["n_redeem_total"] = cr.groupby("household_key").size().reindex(idx).fillna(0)
    cr84 = cr[cr.day > snapshot_day-84]
    out["n_redeem_84"] = cr84.groupby("household_key").size().reindex(idx).fillna(0)
    last = cr.groupby("household_key").day.max().reindex(idx)
    out["days_since_redeem"] = (snapshot_day - last).fillna(9999)
    return out.reset_index()

new = A.build_features(fn)
print(new.shape)
e002 = A.load_saved("e002_mix.parquet")
m = e002.merge(new.drop(columns=["snapshot_day"]), on="household_key", how="left")
print(m.shape)

# demographics + interactions
demo = A.snapshot().table("demographics")
m = m.merge(demo, on="household_key", how="left")
m["has_demo"] = m.classification_1.notna().astype(int)
m["age_num"] = m.classification_1.str.extract(r"(\d+)").astype(float)
m["level_num"] = m.classification_3.str.extract(r"(\d+)").astype(float)
m["size_num"] = m.classification_4.str.extract(r"(\d+)").astype(float)
m["grp_num"] = m.classification_5.str.extract(r"(\d+)").astype(float)
m["kid_num"] = m.kid_category_desc.map({"1":1,"2":2,"3+":3,"None/Unknown":0})
s = m["spend_28"]
m["s28_x_age"] = s*m.age_num.fillna(0)
m["s28_x_size"] = s*m.size_num.fillna(0)
m["s28_x_kid"] = s*m.kid_num.fillna(0)
m["s28_x_wsin"] = s*m.week_sin
m["s28_x_wcos"] = s*m.week_cos
m["s28_x_home"] = s*(m.homeowner_desc=="Homeowner").astype(int)
m["s28_x_tenure"] = s*np.log1p(m.tenure)
print(m.shape)
path = A.save_table(m, "e003_camp_demo_inter")
print(path)
