ct = agent_api.snapshot().table("campaign_targets")
print(ct.shape, ct.duplicated(subset=["household_key","campaign"]).sum(), ct.duplicated().sum())
print(ct["description"].value_counts())
demo = agent_api.snapshot().table("demographics")
print(demo.shape, demo.duplicated(subset=["household_key"]).sum())
# check campaigns table
camp = agent_api.snapshot().table("campaigns")
print(camp.shape, camp.duplicated(subset=["campaign"]).sum())
print(camp.head())
