v = agent_api.snapshot()
dm = v.display_mailer
print(dm.duplicated(subset=["product_id","store_id","week_no"]).sum())
print(dm.product_id.nunique(), dm.store_id.nunique(), dm.week_no.min(), dm.week_no.max())
tx = v.transactions
print(tx.shape, tx.week_no.min(), tx.week_no.max())
