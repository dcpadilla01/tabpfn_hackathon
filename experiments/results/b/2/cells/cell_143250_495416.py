import agent_api, pandas as pd
base = agent_api.load_saved('e001_recent_spend.parquet')
mkt = agent_api.load_saved('e002_marketing.parquet')
m = base.merge(mkt.drop(columns=['household_key','snapshot_day']), left_index=True, right_index=True, how='left')
print(m.shape)
print(m.columns.tolist())
print(m.isna().mean().round(3).to_string())
path = agent_api.save_table(m,'e002_marketing_v2')
print(path)
