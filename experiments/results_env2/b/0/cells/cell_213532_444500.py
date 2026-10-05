base = agent_api.load_saved("e002_mix.parquet")

def add_feats(view, snap):
    hh = view.households
    df = pd.DataFrame(index=hh)
    tx = view.table("transactions")

    # --- marketing exposure (aggregated, no raw merges) ---
    camp = view.table("campaigns")
    started = set(camp["campaign"])
    cmap = camp.set_index("campaign")["start_day"]
    ct = view.table("campaign_targets")
    ct = ct[ct["campaign"].isin(started)]
    g = ct.groupby("household_key")
    df["n_campaigns"] = g.size()
    piv = ct.pivot_table(index="household_key", columns="description",
                         values="campaign", aggfunc="nunique")
    for c in ["TypeA", "TypeB", "TypeC"]:
        df["camp_" + c] = piv[c] if c in piv.columns else 0
    ct2 = ct.assign(start_day=ct["campaign"].map(cmap))
    df["camp_recent84"] = ct2[ct2["start_day"] > snap - 84].groupby("household_key").size()

    cred = view.table("coupon_redemptions")
    cred = cred[cred["day"] <= snap]
    df["redemp_total"] = cred.groupby("household_key").size()
    df["redemp_84"] = cred[cred["day"] > snap - 84].groupby("household_key").size()
    df["days_since_redemp"] = snap - cred.groupby("household_key")["day"].max()

    # --- demographics as ordinal codes ---
    demo = view.table("demographics")
    d = pd.DataFrame(index=demo["household_key"].values)
    d["has_demo"] = 1
    d["age_code"] = demo["classification_1"].str.extract(r"(\d+)").astype(float)
    d["class3_code"] = demo["classification_3"].str.extract(r"(\d+)").astype(float)
    d["size_code"] = demo["classification_4"].replace({"5+": "5"}).astype(float)
    d["class5_code"] = demo["classification_5"].str.extract(r"(\d+)").astype(float)
    d["homeowner_code"] = demo["homeowner_desc"].map(
        {"Homeowner": 4, "Probable Owner": 3, "Probable Renter": 2, "Renter": 1, "Unknown": 0})
    d["kids_code"] = demo["kid_category_desc"].map(
        {"None/Unknown": 0, "1": 1, "2": 2, "3+": 3})
    d["class2"] = demo["classification_2"].astype(str)
    df = df.join(d)

    # --- interactions ---
    df["campA_x_spend28"] = df["camp_TypeA"].fillna(0) * df["spend_28"]
    df["camp_x_demo"] = df["n_campaigns"].fillna(0) * df["has_demo"].fillna(0)
    return df

feats = agent_api.build_features(add_feats)
print("feats:", feats.shape, "dupes:", feats.duplicated(subset=["household_key","snapshot_day"]).sum())
print(feats.isna().mean().round(3).to_string())

out = base.merge(feats, on=["household_key", "snapshot_day"], how="inner")
print("out:", out.shape, "dupes:", out.duplicated(subset=["household_key","snapshot_day"]).sum())
print(out.columns.tolist())
path = agent_api.save_table(out, "e004_marketing_demo")
print(path)
