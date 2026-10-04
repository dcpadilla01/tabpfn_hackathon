import agent_api as api
import pandas as pd, numpy as np

def probe(view, day):
    hh = view.households
    print("DAY", day, "view.day:", view.day, "view.week:", view.week)
    print("hh type:", type(hh), "len:", len(hh) if hh is not None else None)
    print("hh sample:", hh[:5] if hh is not None else None)
    tx = view.transactions
    print("tx:", tx.shape, "maxday:", tx.day.max())
    print("demo:", view.demographics.shape)
    print("ct:", view.campaign_targets.shape)
    return pd.DataFrame(index=hh[:3])

df = api.build_features(probe)
print(df.shape)
