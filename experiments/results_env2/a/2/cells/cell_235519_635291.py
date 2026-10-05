import agent_api as A, pandas as pd, numpy as np

tt = A.train_targets()
oof = A.load_saved('oof_e008.parquet')
d = oof.merge(tt, on=['household_key','snapshot_day'])
print('oof rows', len(d), 'snapshots:', sorted(d.snapshot_day.unique()))
for c in ['oof_sq','oof_med','oof_log']:
    d[f'mae_{c}'] = (d[c]-d.future_spend_4w).abs()
g = d.groupby('snapshot_day').agg(n=('future_spend_4w','size'),
    y_mean=('future_spend_4w','mean'), y_med=('future_spend_4w','median'),
    mae_sq=('mae_oof_sq','mean'), mae_med=('mae_oof_med','mean'), mae_log=('mae_oof_log','mean'))
print(g.round(2))
# optimal per-snapshot multiplicative calibration on OOF (minimize MAE)
for c in ['oof_sq','oof_med','oof_log']:
    facs = {}
    for s, gr in d.groupby('snapshot_day'):
        f = gr.future_spend_4w.values; p = gr[c].values
        fs = np.linspace(0.7,1.3,121)
        maes = [np.abs(f-p*k).mean() for k in fs]
        facs[s] = fs[int(np.argmin(maes))]
    print(c, {k: round(v,3) for k,v in sorted(facs.items())})
