
import pandas as pd, numpy as np
e = load_saved('e011_table.parquet')
tt = train_targets()
print(e.dtypes.value_counts())
df = e.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)
feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
print('nan frac total', X.isna().mean().mean())
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
tr = ~np.isin(days, [403,431]); va = np.isin(days, [403,431])
print('local train', tr.sum(), 'local val', va.sum())

def ridge_eval(cols, alphas=(3,10,30,100,300,1000)):
    Xm = X[cols].fillna(0.0).values
    mu = Xm[tr].mean(0); sd = Xm[tr].std(0)+1e-9
    Z = (Xm-mu)/sd
    Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[va], np.ones(va.sum())]
    ytr = y[tr]
    out=[]
    for a in alphas:
        A = Ztr.T@Ztr + a*np.eye(Ztr.shape[1]); A[-1,-1]-=a
        w = np.linalg.solve(A, Ztr.T@ytr)
        pred = Zva@w
        out.append((a, np.abs(pred-y[va]).mean()))
    return out

base = ridge_eval(feats)
print('E011 all:', base)
# core subset
core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','trips_w1','trips_w2','usual_4w','ratio_recent_usual','tenure_days','gap_mean','gap_std','wk_spend_mean','wk_spend_std','ew_spend_hl14','ew_spend_hl28','ew_spend_hl56','ew_spend_hl112','ya_spend','ya_trips','ya_cov','ratio_ya_28','b_life_spend','b_w_mean6','b_w_cv6','sc_f_mean','sc_f_med','sc_f_ew_hl2','sc_f_ew_hl4','sc_ratio_mean','sc_carry','sc_carry_act','sc_f_zero_frac','sc_f_active_mean','has_demo']
print('core:', ridge_eval(core))
