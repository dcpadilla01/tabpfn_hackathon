v = agent_api.snapshot()
dm = v.display_mailer
print(dm.dtypes)
print("display uniq:", dm.display.unique()[:10])
print("mailer uniq:", dm.mailer.unique()[:10])
