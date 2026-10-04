import pandas as pd, numpy as np, agent_api
e3 = agent_api.load_saved('mkt_v2.parquet')
mix = agent_api.load_saved('mix_v1.parquet')
print('e3', e3.shape, 'mix', mix.shape)
feat_e3 = [c for c in e3.columns if c not in ('household_key','snapshot_day')]
feat_mix = [c for c in mix.columns if c not in ('household_key','snapshot_day')]
print('n feat e3', len(feat_e3), '| n feat mix', len(feat_mix))
print('e3 feats:', feat_e3)
print('mix feats:', feat_mix)
overlap = set(feat_e3) & set(feat_mix)
print('overlap:', overlap)
if overlap:
    mix = mix.rename(columns={c: 'mix_'+c for c in overlap})
df = e3.merge(mix, on=['household_key','snapshot_day'], how='left')
print('merged', df.shape)
print('worst NaN cols:')
print(df.isna().mean().sort_values(ascending=False).head(8))
p = agent_api.save_table(df, 'e005_full_plus_mix.parquet')
print('saved', p)