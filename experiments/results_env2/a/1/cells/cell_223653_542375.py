
import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
NF = agent_api.load_saved('e005_newfeats.parquet')
NF = NF.fillna({'gap_mean':0,'gap_std':0,'gap_max':0,'last_gap':0})
NF['gap_mean']=NF.gap_mean.fillna(0); NF['gap_std']=NF.gap_std.fillna(0); NF['gap_max']=NF.gap_max.fillna(0); NF['last_gap']=NF.last_gap.fillna(0)
D = F.merge(T, on=['household_key','snapshot_day'], how='left').merge(NF, on=['household_key','snapshot_day'], how='left')
newf = [c for c in NF.columns if c not in ('household_key','snapshot_day')]
oldf = [c for c in F.columns if c not in ('household_key','snapshot_day')]
print('nan:', D[newf].isna().sum().sum())
base = dict(n_estimators=2400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

def run(V, feats):
    tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
    w = 0.5 ** ((V - tr.snapshot_day) / 277)
    m = xgb.XGBRegressor(**base); m.fit(tr[feats], tr.future_spend_4w, sample_weight=w, verbose=False)
    pv = np.clip(m.predict(va[feats]), 0, None)
    return np.abs(pv - va.future_spend_4w.values).mean(), m

t0=time.time()
subsets = {
 'all_new': newf,
 'seq+gap': [c for c in newf if c.startswith(('s1','s2','s3','s4','seq','ratio','gap','last_gap'))],
 'dept': [c for c in newf if c.startswith(('dsp','drec','dept28'))],
 'seq+dept': [c for c in newf if c.startswith(('s1','s2','s3','s4','seq','ratio','dsp','drec','dept28'))],
 'wk': [c for c in newf if c.startswith('wk_')],
}
for V in (431, 403):
    m0,_ = run(V, oldf)
    line = f'V={V}: base {m0:.3f} |'
    for nm, sub in subsets.items():
        m1,_ = run(V, oldf+sub)
        line += f' {nm} {m1:.3f} |'
    print(line)
print(f'({time.time()-t0:.0f}s)')
