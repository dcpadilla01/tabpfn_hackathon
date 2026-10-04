t = agent_api.load_saved('rich_behavioral.parquet')
print("shape:", t.shape)
print("columns:", list(t.columns))
print(t.dtypes.value_counts())
print("snapdays:", agent_api.snapshot_days())
s = agent_api.snapshot()
dm = s.display_mailer
print("\ndisplay_mailer shape:", dm.shape)
print(dm.head(3))
print("display vals:", dm['display'].value_counts().head().to_dict())
print("mailer vals:", dm['mailer'].value_counts().head().to_dict())
ct = s.campaign_targets
print("\ncampaign_targets shape:", ct.shape)
print(ct.head(3))
print(ct['description'].value_counts().to_dict())
cr = s.coupon_redemptions
print("\ncoupon_redemptions shape:", cr.shape)
print(cr.head(3))
camps = s.campaigns
print("\ncampaigns shape:", camps.shape)
print(camps.head(5))
