import agent_api as A
print(A.snapshot_days())
print(A.describe_tables())
snap = A.snapshot()
tr = snap.transactions
print(tr.shape)
print(tr.sales_value.describe())
# target distribution
tt = A.train_targets()
print(tt.future_spend_4w.describe())
print((tt.future_spend_4w==0).mean())
