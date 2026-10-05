df = agent_api.load_saved("e001_recent_behavior.parquet")
print(df.shape)
print(df.columns.tolist())
print(df.head())


# ---- cell ----
v = agent_api.snapshot(459)
tx = v.transactions
prod = v.table("products")
print(tx.shape, prod.shape)
m = tx.merge(prod[["product_id","department","brand"]], on="product_id", how="left")
print(m["department"].value_counts().head(20))
print(m["brand"].value_counts(dropna=False))
# spend share by dept overall
g = m.groupby("department")["sales_value"].sum().sort_values(ascending=False)
print(g.head(15))
print("total spend", g.sum())


# ---- cell ----
def fn(view, snapshot_day):
    import numpy as np, pandas as pd
    tx = view.table("transactions")
    prod = view.table("products")
    tx = tx.merge(prod[["product_id","department","brand"]], on="product_id", how="left")
    hh = view.households
    d = snapshot_day
    w = tx[(tx.day > d-84) & (tx.day <= d)]
    tot = w.groupby("household_key")["sales_value"].sum()
    out = pd.DataFrame(index=hh.index)
    out["tot84"] = tot.reindex(hh.index).fillna(0.0)
    # department spend shares over trailing 84d
    gs = w.groupby(["household_key","department"])["sales_value"].sum()
    gs = gs.unstack(fill_value=0.0).reindex(hh.index, fill_value=0.0)
    for col in ["GROCERY","PRODUCE","DRUG GM","DELI","PASTRY"]:
        if col in gs.columns:
            out["sh_"+col[:6].strip().replace(" ","_")] = gs[col]/out["tot84"].replace(0, np.nan)
        else:
            out["sh_"+col[:6].strip().replace(" ","_")] = 0.0
    meat = gs["MEAT"].fillna(0)+gs.get("MEAT-PCKGD", 0)
    out["sh_meat"] = meat/out["tot84"].replace(0, np.nan)
    # private brand share
    pr = w[w.brand=="Private"].groupby("household_key")["sales_value"].sum()
    out["sh_private"] = pr.reindex(hh.index).fillna(0.0)/out["tot84"].replace(0, np.nan)
    # diversity: entropy of dept shares, n depts, top dept share
    gsum = gs.sum(axis=1).replace(0, np.nan)
    p = gs.div(gsum, axis=0)
    ent = -(p.where(p>0)*np.log(p.where(p>0))).sum(axis=1)
    out["dept_entropy"] = ent.fillna(0.0)
    out["n_depts_84"] = (gs>0).sum(axis=1).astype(float)
    out["top_dept_share"] = (gs.max(axis=1)/gsum).fillna(0.0)
    out = out.drop(columns=["tot84"])
    return out

X = agent_api.build_features(fn)
print(X.shape)
print(X.head())
print(X.isna().mean().round(3))
path = agent_api.save_table(X, "e002_mix")
print(path)


# ---- cell ----
def fn(view, snapshot_day):
    import numpy as np, pandas as pd
    tx = view.table("transactions")
    prod = view.table("products")
    tx = tx.merge(prod[["product_id","department","brand"]], on="product_id", how="left")
    hh = pd.Index(view.households)
    d = snapshot_day
    w = tx[(tx.day > d-84) & (tx.day <= d)]
    tot = w.groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0.0)
    out = pd.DataFrame(index=hh)
    out["tot84"] = tot
    gs = w.groupby(["household_key","department"])["sales_value"].sum().unstack(fill_value=0.0).reindex(hh, fill_value=0.0)
    for col in ["GROCERY","PRODUCE","DRUG GM","DELI","PASTRY"]:
        out["sh_"+col[:6].strip().replace(" ","_")] = gs[col]/tot.replace(0, np.nan) if col in gs.columns else 0.0
    meat = gs.get("MEAT", 0)+gs.get("MEAT-PCKGD", 0)
    out["sh_meat"] = meat/tot.replace(0, np.nan)
    pr = w[w.brand=="Private"].groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0.0)
    out["sh_private"] = pr/tot.replace(0, np.nan)
    gsum = gs.sum(axis=1).replace(0, np.nan)
    p = gs.div(gsum, axis=0)
    ent = -(p.where(p>0)*np.log(p.where(p>0))).sum(axis=1)
    out["dept_entropy"] = ent.fillna(0.0)
    out["n_depts_84"] = (gs>0).sum(axis=1).astype(float)
    out["top_dept_share"] = (gs.max(axis=1)/gsum).fillna(0.0)
    return out.drop(columns=["tot84"])

X = agent_api.build_features(fn)
print(X.shape)
print(X.isna().mean().round(3))
path = agent_api.save_table(X, "e002_mix")
print(path)


# ---- cell ----
base = agent_api.load_saved("e001_recent_behavior.parquet")
mix = agent_api.load_saved("e002_mix.parquet")
m = base.merge(mix.drop(columns=[]), on=["household_key","snapshot_day"], how="inner")
print(m.shape)
path = agent_api.save_table(m, "e002_full")
print(path)
