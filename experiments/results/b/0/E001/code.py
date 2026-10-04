import agent_api as api
print(api.snapshot_days())
v = api.snapshot()
print("households type:", type(v.households))
print(v.households.head() if hasattr(v.households, 'head') else v.households[:5])
print("tx shape:", v.transactions.shape)
print(v.transactions.head(3))
print("demo shape:", v.demographics.shape)
t = api.train_targets()
print("targets:", t.shape)
print(t['future_spend_4w'].describe())
b = api.baseline_features()
print("baseline:", b.shape, b.columns.tolist())
print(b.head(3))


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

def probe(view, day):
    print("DAY", day, "week", view.day, view.week)
    hh = view.households
    print("n hh:", len(hh), type(hh))
    print(hh.head() if hasattr(hh,'head') else hh[:5])
    tx = view.transactions
    print("tx shape:", tx.shape, "max day:", tx['day'].max())
    print(tx.head(2))
    print("demo:", view.demographics.shape)
    print("campaign_targets:", view.campaign_targets.shape)
    print("coupons:", view.coupons.shape)
    print("coupon_redemptions:", view.coupon_redemptions.shape)
    print("display_mailer:", view.display_mailer.shape)
    print("campaigns:", view.campaigns.shape)
    print("products:", view.products.shape)
    return pd.DataFrame(index=hh[:3])

df = api.build_features(probe)
print(df.shape)
print(df.head())


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

v = api.snapshot(95)
print("households:", type(v.households), len(v.households))
print(v.households.head() if hasattr(v.households,'head') else v.households[:5])
print("day:", v.day, "week:", v.week)
tx = v.transactions
print("tx:", tx.shape, "maxday", tx.day.max())
print(tx.head(2))
print("demo:", v.demographics.shape)
print("ct:", v.campaign_targets.shape, "red:", v.coupon_redemptions.shape, "coup:", v.coupons.shape)
print("dm:", v.display_mailer.shape, "camp:", v.campaigns.shape)
print("prod:", v.products.shape)


# ---- cell ----
import agent_api as api
v = api.snapshot(95)
print([a for a in dir(v) if not a.startswith('_')])
print(api.describe_tables())


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
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


# ---- cell ----
import agent_api as api
import pandas as pd, numpy as np

def fn(view, day):
    tx = view.transactions
    hh = view.households
    recent = tx[tx['day'] > day - 28]
    g = recent.groupby('household_key').agg(spend28=('sales_value','sum'), trips28=('basket_id','nunique'))
    last = tx.groupby('household_key')['day'].max()
    out = pd.DataFrame(index=pd.Index(hh, name='household_key'))
    out['spend28'] = g['spend28'].reindex(out.index).fillna(0.0)
    out['trips28'] = g['trips28'].reindex(out.index).fillna(0)
    out['recency'] = (day - last).reindex(out.index).fillna(999)
    return out

df = api.build_features(fn)
print(df.shape, df.columns.tolist())
print(df.head())
p = api.save_table(df, 'rfm28')
print(p)
