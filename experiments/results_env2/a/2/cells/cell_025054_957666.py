
import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

def prof_fn(view, snapshot_day):
    tx = view.table("transactions")
    g = tx.groupby("household_key")
    prof = pd.DataFrame({
        "prof_spend_sum": g.sales_value.sum(),
        "prof_baskets": g.basket_id.nunique(),
        "prof_first_day": g.day.min(),
        "prof_last_day": g.day.max(),
        "prof_ndays_active": g.day.nunique(),
        "prof_spend_mean_line": g.sales_value.mean(),
    })
    prof["prof_tenure"] = (snapshot_day - prof.prof_first_day + 1).clip(lower=1)
    prof["prof_spend_rate"] = prof.prof_spend_sum / prof.prof_tenure
    prof["prof_active_frac"] = prof.prof_baskets / prof.prof_tenure
    prof["prof_spend_per_active_day"] = prof.prof_spend_sum / prof.prof_ndays_active.clip(lower=1)
    # decayed spend rate (hl=112) over full history
    tx2 = tx[["household_key","day","sales_value"]].copy()
    tx2["w"] = np.exp(-np.log(2)*(snapshot_day-tx2.day)/112.0)
    d = tx2.groupby("household_key").agg(dec=("w","sum"), decsp=("sales_value", lambda s: 0))
    tx2["decsp"] = tx2.sales_value*tx2.w
    d = tx2.groupby("household_key").agg(dec=("w","sum"), decsp=("decsp","sum"))
    prof["prof_decay_rate"] = d.decsp/d.dec.clip(lower=1e-9)
    # share of spend in last 84d
    s84 = tx[tx.day>snapshot_day-84].groupby("household_key").sales_value.sum()
    prof["prof_share_last84"] = (s84/prof.prof_spend_sum).fillna(0)
    return prof.drop(columns=["prof_first_day","prof_last_day"])

feats = A.build_features(prof_fn)
print(feats.shape, feats.snapshot_day.nunique())
print(feats.head(3))
A.save_table(feats, "feats_prof.parquet")
