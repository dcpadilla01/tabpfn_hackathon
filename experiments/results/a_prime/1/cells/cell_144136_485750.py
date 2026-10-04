import agent_api as api
v = api.snapshot()
print([a for a in dir(v) if not a.startswith('_')])
print(v.campaigns)
ct = v.campaign_targets
demo = v.demographics
print("demo hh:", demo.household_key.nunique())
print("targets with demo:", ct[ct.household_key.isin(demo.household_key)].household_key.nunique())