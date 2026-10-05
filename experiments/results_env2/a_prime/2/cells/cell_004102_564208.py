
import agent_api as A
v = A.snapshot()
p = v.products
print(p.department.value_counts().head(30))
tr = v.transactions
print(tr.shape)
print(tr.head(3))
# check a household's history for gap/window conventions
h = A.history(v.households.iloc[0], as_of_day=459)
print(h[['day','basket_id','sales_value']].tail(5))
