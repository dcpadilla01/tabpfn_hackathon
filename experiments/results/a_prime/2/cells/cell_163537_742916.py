import agent_api, numpy as np, pandas as pd

base = agent_api.load_saved('e009_ewma_longlags.parquet')
new = agent_api.load_saved('e013_new_feats.parquet')

m = base.merge(new, on=['household_key', 'snapshot_day'], how='left')

dem = agent_api.snapshot().demographics.set_index('household_key')
age = dem.classification_1.str.extract(r'(\d+)').astype(float)[0]
size = dem.classification_4.str.extract(r'(\d+)').astype(float)[0]
lvl = dem.classification_3.str.extract(r'(\d+)').astype(float)[0]
grp = dem.classification_5.str.extract(r'(\d+)').astype(float)[0]
kid = dem.kid_category_desc.map({'None/Unknown': 0, '1': 1, '2': 2, '3+': 3})
home = dem.homeowner_desc.map({'Renter': 0, 'Probable Renter': 1, 'Unknown': 2, 'Probable Owner': 3, 'Homeowner': 4})
dnum = pd.DataFrame({'age_n': age, 'size_n': size, 'lvl_n': lvl, 'grp_n': grp, 'kid_n': kid, 'home_n': home})
dnum.index.name = 'household_key'

m = m.merge(dnum.reset_index(), on='household_key', how='left')
for c in ['size_n', 'kid_n', 'age_n', 'home_n']:
    m[c] = pd.to_numeric(m[c], errors='coerce')

m['ix_spend28_size'] = m['spend_28'] * m['size_n']
m['ix_spend28_kid'] = m['spend_28'] * m['kid_n']
m['ix_spend28_age'] = m['spend_28'] * m['age_n']
m['ix_ewma4_size'] = m['ewma_4'] * m['size_n']
m['ix_ewma4_kid'] = m['ewma_4'] * m['kid_n']
m['ix_tlag_size'] = m['tlag_mean'] * m['size_n']
m['ix_spend28_home'] = m['spend_28'] * m['home_n']
m['ix_size_ewma8'] = m['ewma_8'] * m['size_n']
print(m.shape)
agent_api.save_table(m, 'e013_full.parquet')
