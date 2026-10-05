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
