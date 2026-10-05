
import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')

def fn(view, sd):
    tx = view.table('transactions')
    hh = pd.Index(view.households, name='household_key')
    def wsum(lo, hi=None):
        d = tx[tx.day > lo] if hi is None else tx[(tx.day > lo) & (tx.day <= hi)]
        return d.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    f = pd.DataFrame(index=hh)
    f['spend_336'] = wsum(sd - 336)
    f['spend_yago_4w'] = wsum(sd - 392, sd - 364)
    d84 = tx[tx.day > sd - 84].copy(); d84['wk'] = (d84.day + 8) // 7
    f['wk_active_84'] = d84.groupby('household_key')['wk'].nunique().reindex(hh).fillna(0)
    f['maxwk_84'] = d84.groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').max().reindex(hh).fillna(0)
    d112 = tx[tx.day > sd - 112].copy(); d112['wk'] = (d112.day + 8) // 7
    f['wkstd_112'] = d112.groupby(['household_key','wk'])['sales_value'].sum().groupby('household_key').std().reindex(hh).fillna(0)
    for hl in (28, 56, 112):
        w = 0.5 ** ((sd - tx.day) / float(hl))
        f[f'expdecay_{hl}'] = (tx.sales_value * w).groupby(tx.household_key).sum().reindex(hh).fillna(0)
    b = tx.drop_duplicates('basket_id')
    wb = 0.5 ** ((sd - b.day) / 56.0)
    f['expdecay_b56'] = wb.groupby(b.household_key).sum().reindex(hh).fillna(0)
    return f

fb = agent_api.build_features(fn)
print("build_features:", fb.shape)
feats = agent_api.load_saved('feats_v3.parquet')
full = feats.merge(fb.drop(columns=['household_key','snapshot_day'], errors='ignore') if 'snapshot_day' in fb.columns else fb,
                   left_on=['household_key','snapshot_day'], right_index=True, how='inner')
print("full:", full.shape)
agent_api.save_table(full, 'feats_v4.parquet')
