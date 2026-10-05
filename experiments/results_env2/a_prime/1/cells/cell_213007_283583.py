
for d in (95, 459):
    v = agent_api.snapshot(d)
    hh = v.households
    print(d, type(hh), len(hh) if hh is not None else None)
    ct = v.campaign_targets
    print("  campaign_targets:", ct.shape, ct.description.value_counts().to_dict() if len(ct) else {})
    print("  redemptions:", v.coupon_redemptions.shape)
    print("  campaigns:", v.campaigns.shape)
