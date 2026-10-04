import agent_api, numpy as np, pandas as pd
print(agent_api.snapshot_days())
tt = agent_api.train_targets()
print("targets:", tt.shape)
print(tt.future_spend_4w.describe())
snap = agent_api.snapshot()
tr = snap.transactions
print("transactions:", tr.shape)
print(tr.sales_value.describe())
for c in ['coupon_disc','coupon_match_disc','retail_disc','quantity','day']:
    print(c, tr[c].describe().round(2).to_dict())
hh = snap.households
print("households at 459:", len(hh))
print(tr.head(3))


# ---- cell ----
import agent_api, numpy as np, pandas as pd
snap = agent_api.snapshot()
print(type(snap))
print([a for a in dir(snap) if not a.startswith('_')])
print("day:", snap.day, "week:", snap.week)
hh = snap.households
print("households:", type(hh), hh[:10])


# ---- cell ----
import agent_api, numpy as np, pandas as pd
snap = agent_api.snapshot()
tr = snap.transactions
print(tr.dtypes)
print(tr.head(3))
# check one household history
h = agent_api.history(tr.household_key.iloc[0])
print(type(h), h.shape)
print(h.head(5))
print("basket sizes:")
bs = tr.groupby('basket_id').sales_value.sum()
print(bs.describe())


# ---- cell ----
import agent_api, numpy as np, pandas as pd
tt = agent_api.train_targets()
snap = agent_api.snapshot()
tr = snap.transactions

# per (hh, snapshot) last-28d spend and trips, computed for train snapshots
res = []
for sd in agent_api.snapshot_days()['train']:
    v = agent_api.snapshot(as_of_day=sd)
    t = v.transactions
    w = t[t.day > sd-28].groupby('household_key').agg(s=('sales_value','sum'), n=('basket_id','nunique'), last_day=('day','max'))
    idx = pd.DataFrame(index=v.households)
    idx = idx.join(w).fillna({'s':0,'n':0})
    idx['recency'] = sd - idx['last_day']
    idx['snapshot_day']=sd
    res.append(idx[['snapshot_day','s','n','recency']])
f = pd.concat(res).reset_index().rename(columns={'index':'household_key'})
m = tt.merge(f, on=['household_key','snapshot_day'])
print(m[['future_spend_4w','s','n','recency']].corr().round(3))
# by snapshot
print(m.groupby('snapshot_day')[['future_spend_4w','s']].mean().round(1))
# spend28 vs target scatter quantiles
m['bin'] = pd.qcut(m.s, 10, duplicates='drop')
print(m.groupby('bin').agg(t=('future_spend_4w','mean'), s=('s','mean'), n=('n','mean')).round(1))


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def peek(view, sd):
    print("sd:", sd, "households:", type(view.households), len(view.households) if view.households is not None else None)
    print("day:", view.day, "week:", view.week)
    return pd.DataFrame({'x':[1]}, index=view.households[:5])

df = agent_api.build_features(peek)
print(df.head())


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def peek(view, sd):
    print("sd:", sd, "households type:", type(view.households))
    print("first 5:", list(view.households[:5]) if view.households is not None else None)
    print("n:", len(view.households) if view.households is not None else 0)
    print("day:", view.day)
    # transactions inside a snapshot: check max day
    t = view.transactions
    print("txn max day:", t.day.max(), "shape:", t.shape)
    return pd.DataFrame({'x':[1]}, index=view.households[:5])

df = agent_api.build_features(peek)


# ---- cell ----
import agent_api, numpy as np, pandas as pd

out = []
def peek(view, sd):
    out.append((sd, type(view.households), list(view.households[:5]) if view.households is not None else None,
                len(view.households) if view.households is not None else 0, view.day,
                view.transactions.day.max(), view.transactions.shape))
df = agent_api.build_features(peek)
for o in out: print(o)
print(df.head())


# ---- cell ----
import agent_api, numpy as np, pandas as pd

out = []
def peek(view, sd):
    out.append((sd, type(view.households), list(view.households[:5]) if view.households is not None else None,
                len(view.households) if view.households is not None else 0, view.day,
                view.transactions.day.max(), view.transactions.shape))
    return pd.DataFrame({'x':[1]}, index=[1,2,3])

df = agent_api.build_features(peek)
for o in out: print(o)


# ---- cell ----
import agent_api, numpy as np, pandas as pd

out = []
def peek(view, sd):
    out.append((sd, type(view.households), list(view.households[:5]) if view.households is not None else None,
                len(view.households) if view.households is not None else 0, view.day,
                view.transactions.day.max(), view.transactions.shape))
    return pd.DataFrame({'x':[1]}, index=[1,2,3])

df = agent_api.build_features(peek)
print("DONE", len(out))
for o in out: print(o)


# ---- cell ----
import agent_api, numpy as np, pandas as pd

logs = []
def peek(view, sd):
    logs.append((sd, len(view.households), view.day, view.transactions.day.max(), view.transactions.shape[0]))
    return pd.DataFrame({'x':[1]}, index=[1,2,3])

df = agent_api.build_features(peek)
print("DONE", len(logs))
for o in logs: print(o)
