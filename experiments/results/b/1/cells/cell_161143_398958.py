import agent_api as api
t = api.load_saved('cand_new.parquet')
print(t.shape)
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print(len(cols), cols)
# check overlap with e009
e9 = api.load_saved('e009_macro.parquet')
e9c = [c for c in e9.columns if c not in ('household_key','snapshot_day')]
print('overlap with e009:', set(cols)&set(e9c))
# quick: distribution of target-ish? just check a few cand features describe
print(t[['a2','a7','dsl','rvu','ly_spend']].describe().T)