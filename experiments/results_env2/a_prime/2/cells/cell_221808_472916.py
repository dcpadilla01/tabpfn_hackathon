import pandas as pd, numpy as np, agent_api
df = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
d2 = df.merge(tt, on=['household_key','snapshot_day'], how='inner')  # train rows only
feats = [c for c in d2.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
y = d2.future_spend_4w.values

def fit_pred(Xtr, ytr, Xva, a=300):
    mu=np.nanmean(Xtr,axis=0); sg=np.nanstd(Xtr,axis=0)+1e-9
    Ztr=np.nan_to_num((Xtr-mu)/sg); Zva=np.nan_to_num((Xva-mu)/sg)
    w=np.linalg.solve(Ztr.T@Ztr+a*np.eye(Ztr.shape[1]), Ztr.T@(ytr-ytr.mean()))
    return Zva@w+ytr.mean()

# two inner eval snapshots for stability: fit on <=375, eval on 403+431
mfit = d2.snapshot_day<=375; miv = d2.snapshot_day>=403
X = d2[feats].values.astype(float)
base = np.abs(fit_pred(X[mfit.values], y[mfit.values], X[miv.values], 300)-y[miv.values]).mean()
print('inner base (E005 raw, a=300): %.3f  (n_eval=%d)' % (base, miv.sum()))

# candidate engineered features (raw scale)
s28=d2.spend_28.fillna(0); s84=d2.spend_84.fillna(0); s365=d2.spend_365.fillna(0)
dsl=d2.days_since_last.fillna(999); ew84=d2.ew_84.fillna(0); ew28=d2.ew_28.fillna(0)
act=d2.active_28.astype(float); b28=d2.baskets_28.fillna(0); r28=d2.spend_28_ratio.fillna(1)
cands = {
 'dsl_x_ew84': dsl*ew84/100.0,
 'act_x_ew84': act*ew84,
 'act_x_s84': act*s84,
 'sqrt_s84': np.sqrt(s84),
 'sqrt_s28': np.sqrt(s28),
 's84_sq': s84**2/1000.0,
 'dsl_flag28': (dsl>28).astype(float),
 'dsl_flag56': (dsl>56).astype(float),
 'zero_frac_365': 1-np.minimum(1, (s28>0).astype(float)),  # placeholder
 'inv_dsl': 1.0/(1+dsl),
 'ew28_x_b28': ew28*b28,
 's84_minus_s365_4w': s84 - s365/13.0,
}
for k,v in cands.items():
    Xn = np.column_stack([X, v.values.astype(float)])
    m = np.abs(fit_pred(Xn[mfit.values], y[mfit.values], Xn[miv.values], 300)-y[miv.values]).mean()
    print('%-20s %.3f  (%+.3f)' % (k, m, m-base))