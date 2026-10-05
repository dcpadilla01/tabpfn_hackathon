import agent_api, pandas as pd, numpy as np

def probe(view, sd):
    hh = view.households
    print("day, week:", view.day, view.week, "| hh type:", type(hh), "len:", len(hh))
    print("sample hh:", list(hh)[:5])
    tx = view.table("transactions")
    print("tx shape:", tx.shape, "day range:", tx.day.min(), tx.day.max())
    print("tx cols:", list(tx.columns))
    tg = view.table("campaign_targets")
    print("campaign_targets shape:", tg.shape, "cols:", list(tg.columns))
    print("descriptions:", tg.description.value_counts().to_dict() if len(tg) else {})
    cp = view.table("coupon_redemptions")
    print("redemptions shape:", cp.shape)
    dm = view.table("display_mailer")
    print("dm shape:", dm.shape, "week max:", dm.week_no.max())
    return pd.DataFrame(index=hh)

out = agent_api.build_features(probe)
print("out shape:", out.shape)
print("snapshot days:", sorted(out.snapshot_day.unique()))
