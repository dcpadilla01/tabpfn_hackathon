def probe(view, day):
    print("DAY", day, "households", type(view.households), getattr(view.households,'shape',None))
    if view.households is not None:
        print(view.households.head(3))
    tx = view.table("transactions")
    print("tx", tx.shape, tx.day.min(), tx.day.max())
    ct = view.table("campaign_targets")
    print("ct", ct.shape, ct.head(2))
    cr = view.table("coupon_redemptions")
    print("cr", cr.shape)
    dm = view.table("display_mailer")
    print("dm", dm.shape, dm.week_no.max())
    print("week", view.week)
    return view.households.iloc[:2].to_frame("hh") if view.households is not None else None

f = agent_api.build_features(probe)
print(f.shape)
