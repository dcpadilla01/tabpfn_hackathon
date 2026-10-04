import agent_api as api

base = api.load_saved("e018_timing_hazard.parquet")   # E016 best table (157 feats)
demo = api.load_saved("e011_demo.parquet")
v2   = api.load_saved("e017_v2.parquet")

print("base cols with demo-ish names:", [c for c in base.columns if 'class' in c or 'home' in c or 'kid' in c or 'demo' in c])
print("demo table cols:", list(demo.columns))
print("v2 extras:", [c for c in v2.columns if c not in base.columns])
print("base dup rows on keys:", base.duplicated(['household_key','snapshot_day']).sum())
print("demo dup rows on keys:", demo.duplicated(['household_key','snapshot_day']).sum())
print("demo dtypes:", demo.dtypes.to_dict())