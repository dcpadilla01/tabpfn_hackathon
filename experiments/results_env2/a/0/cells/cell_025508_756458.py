
import agent_api as A, numpy as np, pandas as pd
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); oof = A.load_saved('oof_e013.parquet')
cand = A.load_saved('cand_train.parquet')
m = tt.merge(oof, on=['household_key','snapshot_day']).merge(feats, on=['household_key','snapshot_day']).merge(cand, on=['household_key','snapshot_day'])
print('merged', m.shape)
m['resid'] = m.future_spend_4w - m.oof
m['r_7_28'] = m.spend_7/(m.spend_28+1); m['r_28_84'] = m.spend_28/(m.spend_84+1); m['r_84_168'] = m.spend_84/(m.spend_168+1)
m['lag_std'] = m[['lag1_spend','lag2_spend','lag3_spend']].std(axis=1)
W = m[['wk%d'%i for i in range(8)]].values; t8 = np.arange(8)
m['wk_slope'] = ((W*t8).sum(1)*8 - W.sum(1)*t8.sum())/(8*(t8**2).sum()-t8.sum()**2)
base = ['spend_7','spend_28','spend_84','spend_168','lag1_spend','lag2_spend','lag3_spend','exp4w_blend','trips_28','trips_84','tenure','days_since_last','wk_mean8','wk_cv8','active_weeks8','snap_day','snap_cycle_pos']
cands = ['units_84','spend84','trips_7','nprod_28','ncomm_28','up_mean_84','premium_idx','store_rich_84','disp_share_84','mail_share_84','r_7_28','r_28_84','r_84_168','lag_std','wk_slope']
for c in cands+base:
    x = m[c].fillna(m[c].median())
    print('%-16s corr_resid %+.4f  corr_y %+.4f  corr_oof %+.3f' % (c, np.corrcoef(x, m.resid)[0,1], np.corrcoef(x, m.future_spend_4w)[0,1], np.corrcoef(x, m.oof)[0,1]))
print(m[['disp_share_84','mail_share_84','premium_idx','store_rich_84']].describe().round(4))
