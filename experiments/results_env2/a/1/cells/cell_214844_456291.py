
import pandas as pd, numpy as np

def fn(view, snapshot_day):
    d = snapshot_day
    tx = view.table("transactions")
    hh = pd.Index(view.households)
    def spend(win, offset=0):
        m = (tx.day > d - offset - win) & (tx.day <= d - offset)
        return tx[m].groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)
    s28, s56, s84, s168 = spend(28), spend(56), spend(84), spend(168)
    s_ly = spend(28, 364)
    last = tx.groupby("household_key").day.max().reindex(hh)
    rec = (d - last).fillna(999)
    return pd.DataFrame({
        "s28": s28, "s56_2": s56/2, "s84_3": s84/3, "s168_6": s168/6, "s_ly": s_ly,
        "blend": 0.5*s28 + 0.3*s56/2 + 0.2*s84/3,
        "rec": rec,
    }, index=hh)

feats = agent_api.build_features(fn)
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
m = feats.join(tt.future_spend_4w, on=["household_key","snapshot_day"])
print("rows:", len(m), "y nan:", m.future_spend_4w.isna().sum())
for k in ["s28","s56_2","s84_3","s168_6","s_ly","blend"]:
    print("naive MAE", k, round((m[k]-m.future_spend_4w).abs().mean(),3))
for d in sorted(m.snapshot_day.unique()):
    sub = m[m.snapshot_day==d]
    print(d, "corr:", {k: round(np.corrcoef(sub[k], sub.future_spend_4w)[0,1],3) for k in ["s28","s56_2","s84_3","s_ly","rec"]},
          "zero_y:", round((sub.future_spend_4w==0).mean(),3), "zero_s28:", round((sub.s28==0).mean(),3))
