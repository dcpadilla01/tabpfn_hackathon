import pandas as pd, agent_api
df = agent_api.load_saved("e001_history.parquet")
print(df.columns.tolist())
print(df.shape)
print(df.head(3))
print(agent_api.snapshot_days())


# ---- cell ----
import pandas as pd, numpy as np, agent_api

def build(view, day):
    tx = view.table("transactions")
    prod = view.table("products")
    hh = pd.Index(view.households)
    m = tx.merge(prod[["product_id","department","brand"]], on="product_id", how="left")
    m["department"] = m["department"].fillna("UNK")
    m["brand"] = m["brand"].fillna("UNK")

    wins = {}
    for w in (14,28,56,84,112,182,364):
        wins[w] = m[m.day > day-w]

    res = {}
    for w in (14,28,56,84,112,182,364):
        res[f"spend_{w}"] = wins[w].groupby("household_key").sales_value.sum()
    for w in (28,56,84,364):
        res[f"trips_{w}"] = wins[w].groupby("household_key").basket_id.nunique()
    for w in (28,84):
        res[f"qty_{w}"] = wins[w].groupby("household_key").quantity.sum()

    # same 4-week window one year earlier
    ly = m[(m.day > day-392) & (m.day <= day-364)]
    res["spend_ly"] = ly.groupby("household_key").sales_value.sum()

    df = pd.DataFrame(res).reindex(hh)

    # derived spend dynamics
    df["trend"] = df["spend_28"] - df["spend_56"]/2
    df["spend_28_56"] = df["spend_56"] - df["spend_28"]
    df["spend_84_28"] = df["spend_84"] - df["spend_28"]
    df["no_trip_28"] = (df["trips_28"].fillna(0)==0).astype(int)
    df["no_trip_56"] = (df["trips_56"].fillna(0)==0).astype(int)

    # basket composition over 84d
    w84 = wins[84]
    g = w84.groupby("household_key")
    lines = g.size().rename("lines_84")
    df = df.join(lines)
    df["avg_basket_84"] = df["spend_84"]/df["trips_84"].replace(0,np.nan)
    df["lines_per_trip"] = df["lines_84"]/df["trips_84"].replace(0,np.nan)
    df["n_products_84"] = g.product_id.nunique()
    df["n_depts_84"] = g.department.nunique()
    df["unit_price_84"] = df["spend_84"]/df["qty_84"].replace(0,np.nan)

    # brand + discount behaviour
    priv = w84[w84.brand=="National"].groupby("household_key").sales_value.sum()
    df["national_share_84"] = priv/df["spend_84"].replace(0,np.nan)
    for c in ("coupon_disc","retail_disc","coupon_match_disc"):
        df[c+"_84"] = w84.groupby("household_key")[c].sum().reindex(hh)
    df["disc_share_84"] = (df["retail_disc_84"].abs()+df["coupon_disc_84"].abs())/df["spend_84"].replace(0,np.nan)

    # shopping pattern
    wknd = w84.assign(wk=(w84.day%7).isin([5,6])).groupby("household_key").wk.mean()
    df["weekend_share_84"] = wknd
    hr = (w84.trans_time//100).fillna(12)
    df["evening_share_84"] = hr.ge(17).groupby(w84.household_key).mean()
    st = w84.groupby(["household_key","store_id"]).sales_value.sum()
    df["top_store_share_84"] = st.groupby(level=0).max()/df["spend_84"].replace(0,np.nan)

    # recency / lifetime
    df["days_since_last"] = day - m.groupby("household_key").day.max()
    df["tenure"] = day - m.groupby("household_key").day.min() + 1
    df["life_spend"] = m.groupby("household_key").sales_value.sum()
    df["spend_rate"] = df["life_spend"]/df["tenure"]

    # department mix shares (84d), global top-10 departments
    top = w84.groupby("department").sales_value.sum().nlargest(10).index
    dv = w84[w84.department.isin(top)].pivot_table(index="household_key", columns="department",
                                                   values="sales_value", aggfunc="sum")
    dv = dv.reindex(columns=top).reindex(hh)
    dv = dv.div(df["spend_84"].replace(0,np.nan), axis=0)
    dv.columns = [f"dept_{c[:12]}" for c in top]
    df = df.join(dv)
    df["dept_other"] = (1 - dv.sum(axis=1)).clip(0,1)

    # calendar
    week = (day+8)//7
    df["week"] = week
    df["week_sin"] = np.sin(2*np.pi*week/52); df["week_cos"] = np.cos(2*np.pi*week/52)
    df["week_q"] = week % 13
    df["snapshot_day"] = day
    return df

feats = agent_api.build_features(build)
print("shape", feats.shape, "nfeat", feats.shape[1]-2)

tt = agent_api.train_targets()
tr = feats.merge(tt, on=["household_key","snapshot_day"])
num = tr.select_dtypes(include=[np.number]).drop(columns=["snapshot_day"])
cor = num.corr()["future_spend_4w"].drop("future_spend_4w").sort_values(key=np.abs, ascending=False)
print(cor.round(3).to_string())

path = agent_api.save_table(feats, "e002_mix.parquet")
print(path)


# ---- cell ----
import pandas as pd, numpy as np, agent_api

def build(view, day):
    tx = view.table("transactions")
    prod = view.table("products")
    hh = pd.Index(view.households)
    prod["department"] = prod["department"].astype(str)
    prod["brand"] = prod["brand"].astype(str)
    m = tx.merge(prod[["product_id","department","brand"]], on="product_id", how="left")
    m["department"] = m["department"].fillna("UNK")
    m["brand"] = m["brand"].fillna("UNK")

    wins = {}
    for w in (14,28,56,84,112,182,364):
        wins[w] = m[m.day > day-w]

    res = {}
    for w in (14,28,56,84,112,182,364):
        res[f"spend_{w}"] = wins[w].groupby("household_key").sales_value.sum()
    for w in (28,56,84,364):
        res[f"trips_{w}"] = wins[w].groupby("household_key").basket_id.nunique()
    for w in (28,84):
        res[f"qty_{w}"] = wins[w].groupby("household_key").quantity.sum()

    ly = m[(m.day > day-392) & (m.day <= day-364)]
    res["spend_ly"] = ly.groupby("household_key").sales_value.sum()

    df = pd.DataFrame(res).reindex(hh)

    df["trend"] = df["spend_28"] - df["spend_56"]/2
    df["spend_28_56"] = df["spend_56"] - df["spend_28"]
    df["spend_84_28"] = df["spend_84"] - df["spend_28"]
    df["no_trip_28"] = (df["trips_28"].fillna(0)==0).astype(int)
    df["no_trip_56"] = (df["trips_56"].fillna(0)==0).astype(int)

    w84 = wins[84]
    g = w84.groupby("household_key")
    lines = g.size().rename("lines_84")
    df = df.join(lines)
    df["avg_basket_84"] = df["spend_84"]/df["trips_84"].replace(0,np.nan)
    df["lines_per_trip"] = df["lines_84"]/df["trips_84"].replace(0,np.nan)
    df["n_products_84"] = g.product_id.nunique()
    df["n_depts_84"] = g.department.nunique()
    df["unit_price_84"] = df["spend_84"]/df["qty_84"].replace(0,np.nan)

    nat = w84[w84.brand=="National"].groupby("household_key").sales_value.sum()
    df["national_share_84"] = nat/df["spend_84"].replace(0,np.nan)
    for c in ("coupon_disc","retail_disc","coupon_match_disc"):
        df[c+"_84"] = w84.groupby("household_key")[c].sum().reindex(hh)
    df["disc_share_84"] = (df["retail_disc_84"].abs()+df["coupon_disc_84"].abs())/df["spend_84"].replace(0,np.nan)

    wknd = w84.assign(wk=(w84.day%7).isin([5,6])).groupby("household_key").wk.mean()
    df["weekend_share_84"] = wknd
    hr = (w84.trans_time//100).fillna(12)
    df["evening_share_84"] = hr.ge(17).groupby(w84.household_key).mean()
    st = w84.groupby(["household_key","store_id"]).sales_value.sum()
    df["top_store_share_84"] = st.groupby(level=0).max()/df["spend_84"].replace(0,np.nan)

    df["days_since_last"] = day - m.groupby("household_key").day.max()
    df["tenure"] = day - m.groupby("household_key").day.min() + 1
    df["life_spend"] = m.groupby("household_key").sales_value.sum()
    df["spend_rate"] = df["life_spend"]/df["tenure"]

    top = w84.groupby("department").sales_value.sum().nlargest(10).index
    dv = w84[w84.department.isin(top)].pivot_table(index="household_key", columns="department",
                                                   values="sales_value", aggfunc="sum")
    dv = dv.reindex(columns=top).reindex(hh)
    dv = dv.div(df["spend_84"].replace(0,np.nan), axis=0)
    dv.columns = [f"dept_{c[:12]}" for c in top]
    df = df.join(dv)
    df["dept_other"] = (1 - dv.sum(axis=1)).clip(0,1)

    week = (day+8)//7
    df["week"] = week
    df["week_sin"] = np.sin(2*np.pi*week/52); df["week_cos"] = np.cos(2*np.pi*week/52)
    df["week_q"] = week % 13
    df["snapshot_day"] = day
    return df

feats = agent_api.build_features(build)
print("shape", feats.shape, "nfeat", feats.shape[1]-2)

tt = agent_api.train_targets()
tr = feats.merge(tt, on=["household_key","snapshot_day"])
num = tr.select_dtypes(include=[np.number]).drop(columns=["snapshot_day"])
cor = num.corr()["future_spend_4w"].drop("future_spend_4w").sort_values(key=np.abs, ascending=False)
print(cor.round(3).to_string())

path = agent_api.save_table(feats, "e002_mix.parquet")
print(path)
