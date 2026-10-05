
import numpy as np, pandas as pd

def features(view, snapshot_day):
    d = int(snapshot_day)
    tx = view.table("transactions")
    hh = pd.Index(view.households)
    tx28 = tx[tx["day"] > d - 28]
    tx84 = tx[tx["day"] > d - 84]
    txprior = tx[(tx["day"] > d - 112) & (tx["day"] <= d - 28)]

    # per household-product all-time aggregates
    hp = tx.groupby(["household_key", "product_id"]).agg(
        stot=("sales_value", "sum"), nocc=("day", "size"),
        firstd=("day", "min"), lastd=("day", "max"))
    s84 = tx84.groupby(["household_key", "product_id"])["sales_value"].sum().rename("s84")
    s28 = tx28.groupby(["household_key", "product_id"])["sales_value"].sum().rename("s28")
    hp = hp.join(s84).join(s28)
    hp["s84"] = hp["s84"].fillna(0.0); hp["s28"] = hp["s28"].fillna(0.0)

    tot84 = tx84.groupby("household_key")["sales_value"].sum()
    tot28 = tx28.groupby("household_key")["sales_value"].sum()

    hp = hp.reset_index()
    hp["rank84"] = hp.groupby("household_key")["s84"].rank(ascending=False, method="first")
    top = hp[hp["rank84"] <= 10].copy()
    span = np.maximum((d - top["firstd"] + 1) / 28.0, 1.0)
    top["exp28"] = top["stot"] / span                      # all-time avg 28d spend on product
    top["dsl"] = d - top["lastd"]                          # days since last purchase
    top["decay"] = np.exp(-top["dsl"] / 28.0)
    top["exp_dec"] = top["exp28"] * top["decay"]

    agg = top.groupby("household_key").agg(
        staple_exp28=("exp28", "sum"),
        staple_s84=("s84", "sum"),
        staple_s28=("s28", "sum"),
        staple_dsl_med=("dsl", "median"),
        staple_dsl_max=("dsl", "max"),
        staple_active28=("dsl", lambda s: float((s <= 28).sum())),
        staple_n=("exp28", "size"),
    )
    agg["staple_exp_dec"] = top.groupby("household_key")["exp_dec"].sum()
    agg["staple_mom"] = agg["staple_s28"] / agg["staple_exp28"].replace(0.0, np.nan)

    # product concentration in trailing 84d
    hp84 = hp[hp["s84"] > 0].copy()
    tby = hp84.groupby("household_key")["s84"].transform("sum")
    hp84["w"] = (hp84["s84"] / tby.replace(0.0, np.nan)) ** 2
    hhi = hp84.groupby("household_key")["w"].sum().rename("prod_hhi_84")
    nprod84 = hp84.groupby("household_key").size().rename("n_prod_84")
    nprod28 = tx28.groupby("household_key")["product_id"].nunique().rename("n_prod_28")

    # repeat-purchase share: last-28d spend on products bought in prior (28d,112d] window
    p28 = tx28[["household_key", "product_id", "sales_value"]]
    pp = txprior[["household_key", "product_id"]].drop_duplicates()
    m = p28.merge(pp, on=["household_key", "product_id"], how="inner")
    rep = m.groupby("household_key")["sales_value"].sum().rename("rep_spend_28")

    base = pd.DataFrame(index=hh)
    for s in [agg["staple_exp28"], agg["staple_exp_dec"], agg["staple_s84"], agg["staple_s28"],
              agg["staple_mom"], agg["staple_dsl_med"], agg["staple_dsl_max"],
              agg["staple_active28"], agg["staple_n"], hhi, nprod84, nprod28, rep,
              tot84.rename("t84"), tot28.rename("t28")]:
        base[s.name] = s
    base["staple_share84"] = base["staple_s84"] / base["t84"].replace(0.0, np.nan)
    base["repeat_share_28"] = base["rep_spend_28"] / base["t28"].replace(0.0, np.nan)
    base = base.drop(columns=["t84", "t28", "rep_spend_28"])
    return base

df = agent_api.build_features(features)
print(df.shape)
print(df.head(3).T)
print(df.isna().mean().round(3))
path = agent_api.save_table(df, "e007_staples")
print("SAVED", path)
