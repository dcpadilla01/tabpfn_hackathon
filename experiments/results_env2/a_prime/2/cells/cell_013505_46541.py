
import pandas as pd, numpy as np
e019 = agent_api.load_saved('e019_e017_plus_marketing.parquet')
print('e019 shape', e019.shape)
cols19 = [c for c in e019.columns if c not in ('household_key','snapshot_day')]
print('e019 n_feat', len(cols19))
e017 = agent_api.load_saved('e017_xsec_rank.parquet')
cols17 = [c for c in e017.columns if c not in ('household_key','snapshot_day')]
print('e017 n_feat', len(cols17))
adds = [c for c in cols19 if c not in cols17]
print('e019 adds vs e017:', adds)
e018 = agent_api.load_saved('e018_tree_feats.parquet')
cols18 = [c for c in e018.columns if c not in ('household_key','snapshot_day')]
print('e018 feats:', cols18)
print('e018 feats NOT in e019:', [c for c in cols18 if c not in cols19])
print('dtypes of adds:', {c: str(e019[c].dtype) for c in cols19 if e019[c].dtype == object})
tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt['future_spend_4w'].describe())
print('zero share train:', (tt.future_spend_4w == 0).mean())
m = e019.merge(tt, on=['household_key', 'snapshot_day'])
print('merged', m.shape)
nan_share = m[cols19].isna().mean().sort_values(ascending=False)
print('worst NaN coverage:'); print(nan_share.head(12))
corr = m[cols19].corrwith(m['future_spend_4w']).abs().sort_values(ascending=False)
print('top |corr| with target:'); print(corr.head(25))
print('snapshot_days:', agent_api.snapshot_days())
