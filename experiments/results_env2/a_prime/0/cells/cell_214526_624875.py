import pandas as pd, numpy as np

def fn(view, snapshot_day):
    ct = view.table("campaign_targets"); cp = view.table("campaigns")
    cr = view.table("coupon_redemptions"); dm = view.table("display_mailer")
    tx = view.table("transactions")
    try:
        e3 = agent_api.load_saved("e003_union.parquet")
        ok = e3.shape
    except Exception as e:
        ok = "FAIL " + type(e).__name__
    print(f"day={snapshot_day} ct={len(ct)} cp={len(cp)} cr_maxday={cr.day.max() if len(cr) else -1} "
          f"dm_maxweek={dm.week_no.max()} tx_maxday={tx.day.max()} load_saved={ok}")
    return pd.DataFrame(index=view.households[:5])

out = agent_api.build_features(fn)
print("out shape", out.shape)
