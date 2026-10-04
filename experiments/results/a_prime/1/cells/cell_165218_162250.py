import numpy as np, pandas as pd
v = snapshot()
camp = v.campaigns; ct = v.campaign_targets
# For each snapshot day, campaigns overlapping next-28d window and their types
for sd in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    if sd > 459: continue
    act = camp[(camp.start_day <= sd+28) & (camp.end_day >= sd+1)]
    t = ct[ct.campaign.isin(act.campaign)]
    bytype = t.groupby('description')['household_key'].nunique().to_dict()
    print(sd, "camps:", sorted(act.campaign.tolist()), "types:", {k:int(x) for k,x in bytype.items()})