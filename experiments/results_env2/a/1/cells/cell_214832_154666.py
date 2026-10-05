
import pandas as pd, numpy as np

tt = agent_api.train_targets()
y = tt.future_spend_4w
print("train rows:", len(tt), "| zero frac:", (y==0).mean().round(3), "| mean:", y.mean().round(2), "| median:", y.median().round(2), "| p90:", y.quantile(.9).round(2), "| max:", y.max().round(1))

f = agent_api.load_saved("e002_features.parquet")
print("e002_features cols:", f.columns.tolist())
print("e002 shape:", f.shape)

days = agent_api.snapshot_days()["train"]
res = {}
for d in days:
    v = agent_api.snapshot(as_of_day=d)
    tx = v.table("transactions")
    hh = pd.Index(v.households)
    def spend(win, offset=0):
        m = (tx.day > d - offset - win) & (tx.day <= d - offset)
        return tx[m].groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)
    s28, s56, s84, s168 = spend(28), spend(56), spend(84), spend(168)
    s_ly = spend(28, 364)  # same 4-week block one year earlier
    last = tx.groupby("household_key").day.max().reindex(hh)
    rec = (d - last).fillna(999)
    yt = tt[tt.snapshot_day==d].set_index("household_key").future_spend_4w
    idx = yt.index
    preds = {
        "s28": s28.reindex(idx),
        "s56/2": (s56/2).reindex(idx),
        "s84/3": (s84/3).reindex(idx),
        "s168/6": (s168/6).reindex(idx),
        "s_ly": s_ly.reindex(idx),
        "blend": (0.5*s28 + 0.3*s56/2 + 0.2*s84/3).reindex(idx),
    }
    for k,p in preds.items():
        res.setdefault(k, []).append((p - yt).abs().mean())
    if d == days[-1]:
        df = pd.DataFrame({"y": yt, "s28": s28.reindex(idx), "s84": s84.reindex(idx), "s_ly": s_ly.reindex(idx), "rec": rec.reindex(idx)})
        print("corr with y @day", d, ":", df.corr().y.round(3).to_dict())
        print("zero frac of y @", d, (yt==0).mean().round(3), "| zero frac of s28:", (s28.reindex(idx)==0).mean().round(3))
for k, v_ in res.items():
    print("naive MAE", k, round(float(np.mean(v_)), 3))
print("n hh per snapshot:", len(agent_api.snapshot(as_of_day=459).households))
