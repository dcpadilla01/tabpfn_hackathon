import agent_api as A, pandas as pd, numpy as np

def partial_corr(x, y, z):
    # corr of residuals of x~z and y~z (z = list of controls)
    Z = np.column_stack([np.ones(len(z))] + [z[c].values for c in z.columns])
    def res(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = res(x.values.astype(float)), res(y.values.astype(float))
    return np.corrcoef(rx, ry)[0,1]

tt = A.train_targets()
tr = X.merge(tt, on=['household_key','snapshot_day'])
base = A.load_saved("temporal_structure.parquet")
trb = tr.merge(base.drop(columns=[c for c in base.columns if c.startswith(('spend_w','trips_w','lines_w','qty_w','disc_w')) or c in ('basket_mean_w56','basket_max_w56','stores_w84','spend_prev28','spend_ratio_28_56')], errors='ignore'), on=['household_key','snapshot_day'], suffixes=('','_b'))
print(trb.shape)
y = tr.future_spend_4w
ctrl = trb[['spend_28','spend_56','spend_84','trips_28','days_since_last']].copy()
for c in ['disp_spend_28','mail_spend_28','disp_share_28','mail_share_28','disp_lines_28','n_camp_active','n_camp_recent','camp_TypeA_recent','ever_red','red_cnt_56']:
    print(f"{c:18s} partial={partial_corr(tr[c].fillna(0), y, ctrl):+.4f}")
