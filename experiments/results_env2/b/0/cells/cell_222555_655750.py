import agent_api as api
v = api.snapshot()
for name in ["campaigns","campaign_targets","transactions"]:
    t = v.table(name)
    print(name, dict(t.dtypes.astype(str)))
