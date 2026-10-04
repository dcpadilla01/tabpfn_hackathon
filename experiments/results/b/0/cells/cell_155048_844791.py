import pandas as pd, numpy as np

tt = agent_api.train_targets()
y = tt.future_spend_4w
print("train rows", len(tt), "mean", y.mean(), "median", y.median(), "zero share", (y==0).mean())
print(y.describe())

def probe(view, sd):
    hh = view.households
    print("sd", sd, "hh type", type(hh), "len", len(hh))
    if sd == 95:
        print("hh head:", list(hh)[:5])
        t = view.transactions
        print("txn max day", int(t.day.max()), "dm max week", int(view.display_mailer.week_no.max()))
        print("week attr", view.week, "computed wk", (sd+8)//7)
        print("hh indexable?", hh[0] if hasattr(hh,'__getitem__') else 'no getitem')
    return pd.DataFrame({"x": np.arange(len(hh), dtype=float)}, index=hh)

bf = agent_api.build_features(probe)
print("build_features rows", len(bf), "cols", list(bf.columns))
print(bf.groupby('snapshot_day').size())