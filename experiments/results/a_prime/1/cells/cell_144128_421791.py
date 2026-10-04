import agent_api as api
v = api.snapshot()
ct = v.campaign_targets
print(ct.groupby('household_key')['description'].nunique().value_counts().head())
print("hh with any target:", ct.household_key.nunique(), "of", v.households.shape[0])
# overlap with demographics
demo = v.demographics
print("demo hh:", demo.household_key.nunique())
print("targets with demo:", ct[ct.household_key.isin(demo.household_key)].household_key.nunique())
# campaigns started by day 459
print(v.campaigns.head(20))