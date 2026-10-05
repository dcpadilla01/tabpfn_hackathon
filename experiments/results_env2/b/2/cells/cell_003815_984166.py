
import pandas as pd, numpy as np
comb = load_saved('e019_table.parquet')
tt = train_targets()
df = comb.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats = [c for c in comb.columns if c not in ('household_key','snapshot_day')]
NEW = ['x_spend_7d','x_spend_14d','x_trips_7d','x_trips_14d','x_lvl_ew2','x_f_ew_hl6','x_f_ew_hl8',
       'x_f56_mean','x_f84_mean','x_r_std','x_r_ew_hl8','x_zero_streak','x_active_frac13',
       'x_p_s28_rmean','x_p_s28_rmed','x_p_lvl_rmean','x_p_s28_fmean','x_p_usual_rmean','x_p_s28_carry','x_p_ya_rmean']
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
X = df[feats].apply(pd.to_numeric, errors='coerce')
def loso_mae(cols, alpha=1000, pairs=((403,431),(151,179),(263,291),(319,347))):
    Xm = X[list(cols)].fillna(0.0).values
    mu = Xm.mean(0); sd = Xm.std(0)+1e-9
    Z = (Xm-mu)/sd
    maes=[]
    for pair in pairs:
        held = np.isin(days, list(pair)); tr = ~held
        Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[held], np.ones(held.sum())]
        A = Ztr.T@Ztr + alpha*np.eye(Ztr.shape[1]); A[-1,-1]-=alpha
        w = np.linalg.solve(A, Ztr.T@y[tr])
        maes.append(np.abs(Zva@w - y[held]).mean())
    return np.mean(maes), maes
core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','tr wrong']
