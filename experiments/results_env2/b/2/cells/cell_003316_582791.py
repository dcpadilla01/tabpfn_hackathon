
import pandas as pd, numpy as np
e = load_saved('e011_table.parquet')
tt = train_targets()
df = e.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
deal = load_saved('deal_v1.parquet'); haz = load_saved('hazard_v1.parquet'); tim = load_saved('timing_v1.parquet'); disp = load_saved('display_v1.parquet')
blocks={}
for nm,t in [('deal',deal),('haz',haz),('tim',tim),('disp',disp)]:
    t = t.merge(df[['household_key','snapshot_day']], on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
    assert (t.household_key.values==df.household_key.values).all()
    bc = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    blocks[nm]=t[bc]

def loso_mae(Xdf, cols, alpha=1000, pairs=((403,431),(151,179),(263,291),(319,347))):
    Xm = Xdf[list(cols)].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
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

core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','trips_w1','trips_w2','usual_4w','ratio_recent_usual','tenure_days','gap_mean','gap_std','wk_spend_mean','wk_spend_std','ew_spend_hl14','ew_spend_hl28','ew_spend_hl56','ew_spend_hl112','ya_spend','ya_trips','ya_cov','ratio_ya_28','b_life_spend','b_w_mean6','b_w_cv6','sc_f_mean','sc_f_med','sc_f_ew_hl2','sc_f_ew_hl4','sc_ratio_mean','sc_carry','sc_carry_act','sc_f_zero_frac','sc_f_active_mean','has_demo']
m0,_ = loso_mae(df[feats], feats); print('E011 all161 %.3f' % m0)
m1,_ = loso_mae(df[feats], core); print('core48 %.3f' % m1)
base = df[feats]
for bn in blocks:
    d2 = pd.concat([base, blocks[bn]], axis=1)
    m,_ = loso_mae(d2, core+list(blocks[bn].columns)); print(f'core48+{bn}({blocks[bn].shape[1]}) %.3f' % m)
allb = pd.concat([base]+list(blocks.values()), axis=1)
m,_ = loso_mae(allb, core+sum([list(b.columns) for b in blocks.values()],[]))
print('core48+all4blocks %.3f' % m)
# also full E011 + all 4 blocks
m,_ = loso_mae(allb, feats+sum([list(b.columns) for b in blocks.values()],[]))
print('E011+all4blocks %.3f' % m)
