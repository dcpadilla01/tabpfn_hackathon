
import pandas as pd, numpy as np
e = load_saved('e011_table.parquet')
tt = train_targets()
df = e.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key'])
feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def loso_mae(cols, alpha=1000, pairs=None):
    Xm = X[list(cols)].fillna(0.0).values
    mu = Xm.mean(0); sd = Xm.std(0)+1e-9
    Z = (Xm-mu)/sd
    if pairs is None:
        pairs = [((403,431),), ((151,179),), ((263,291),), ((319,347),)]
    maes=[]
    for pair in pairs:
        held = np.isin(days, list(pair))
        tr = ~held
        Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[held], np.ones(held.sum())]
        A = Ztr.T@Ztr + alpha*np.eye(Ztr.shape[1]); A[-1,-1]-=alpha
        w = np.linalg.solve(A, Ztr.T@y[tr])
        maes.append(np.abs(Zva@w - y[held]).mean())
    return np.mean(maes), maes

# candidate extra blocks
deal = load_saved('deal_v1.parquet'); haz = load_saved('hazard_v1.parquet'); tim = load_saved('timing_v1.parquet'); disp = load_saved('display_v1.parquet')
for t in (deal,haz,tim,disp):
    t.sort_values(['snapshot_day','household_key'], inplace=True)
assert (deal.household_key.values==df.household_key.values).all() and (deal.snapshot_day.values==df.snapshot_day.values).all()

blocks = {
 'deal': ['deal_share_84','coup_share_84','deal_line_frac_84','deep_deal_share_84','coup_trips_84','redemp_84','deal_share_364','coup_share_364','deal_line_frac_364','deep_deal_share_364','coup_trips_364','redemp_364','deal_trend'],
 'haz': ['h_gap_mean','h_gap_med','h_gap_std','h_gap_max','h_gap_p90','h_n_gaps','h_gap_last3','h_gap_ratio','h_days_since','h_hazard','h_zero_stretch_112','h_trips_7','h_trips_14','h_active_frac_84'],
 'tim': ['t_hour_mean','t_hour_std','t_morning_sp_share','t_evening_sp_share','t_dow_hhi','t_n_dow','t_n_stores','t_store_share'],
 'disp': ['disp_share_84','mail_share_84','mailAD_share_84','disp_line_frac_84','disp_share_364','mail_share_364','mailAD_share_364','disp_line_frac_364','disp_trend'],
}
m0,_ = loso_mae(feats); print('E011 all %.3f' % m0)
core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','trips_w1','trips_w2','usual_4w','ratio_recent_usual','tenure_days','gap_mean','gap_std','wk_spend_mean','wk_spend_std','ew_spend_hl14','ew_spend_hl28','ew_spend_hl56','ew_spend_hl112','ya_spend','ya_trips','ya_cov','ratio_ya_28','b_life_spend','b_w_mean6','b_w_cv6','sc_f_mean','sc_f_med','sc_f_ew_hl2','sc_f_ew_hl4','sc_ratio_mean','sc_carry','sc_carry_act','sc_f_zero_frac','sc_f_active_mean','has_demo']
m1,_ = loso_mae(core); print('core48 %.3f' % m1)
for bn, bc in blocks.items():
    m,_ = loso_mae(core+bc); print(f'core48+{bn} %.3f' % m)
mall,_ = loso_mae(core+deal+haz+tim+disp, )
