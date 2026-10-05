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
