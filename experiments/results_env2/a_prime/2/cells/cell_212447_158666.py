import agent_api as A
print(A.snapshot_days())
print(A.KEYS, A.TARGET)
v = A.snapshot()
t = v.table("transactions")
print(t.shape)
print(t.sales_value.describe())
hh = t.household_key.nunique()
print("households", hh)
# target distribution on train
tt = A.train_targets()
print(tt.shape)
print(tt.future_spend_4w.describe())
print("zero frac", (tt.future_spend_4w==0).mean())
