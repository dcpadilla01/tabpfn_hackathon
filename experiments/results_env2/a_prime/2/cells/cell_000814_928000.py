
import numpy as np, pandas as pd

BASE = load_saved('e011_discounts.parquet')
base_feats = [c for c in BASE.columns if c not in ('household_key','snapshot_day')]
print('index col sample:', BASE['index'].describe().to_dict())
print('snap_day sample:', BASE['snap_day'].describe().to_dict())

tt = train_targets()
df = BASE.merge(tt, on=['household_key','snapshot_day'])

mom = ['mom28_delta','mom28_ratio','recent_vs_typ','accel','slope_13w','wcv_26']
cal = ['snap_day','wk_of_year','ann_sin','ann_cos','index']

def prep(d, fs):
    X = d[fs].copy()
    for c in fs:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.astype(float); X[c] = X[c].where(X[c]>=0, np.nan)
    return X.astype(float)

def pseudo(fs, lam=30, test_day=431):
    fit_days = [d for d in snapshot_days()['train'] if d < test_day]
    tr = df[df.snapshot_day.isin(fit_days)]; te = df[df.snapshot_day==test_day]
    Xtr, Xte = prep(tr, fs), prep(te, fs)
    mu, sg = Xtr.mean(), Xtr.std().replace(0,1)+1e-9
    Xtr = ((Xtr-mu)/sg).fillna(0.).values; Xte = ((Xte-mu)/sg).fillna(0.).values
    ytr, yte = tr.future_spend_4w.values, te.future_spend_4w.values
    A_ = np.hstack([Xtr, np.ones((len(Xtr),1))]); Av = np.hstack([Xte, np.ones((len(Xte),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    return np.abs(Av@w-yte).mean()

sets = {
 'base78': base_feats,
 'drop_mom_only': [c for c in base_feats if c not in mom],
 'drop_cal_only': [c for c in base_feats if c not in cal],
 'drop_mom+cal': [c for c in base_feats if c not in mom+cal],
 'drop_index_snapday': [c for c in base_feats if c not in ['index','snap_day']],
 'drop_wkofyr_ann': [c for c in base_feats if c not in ['wk_of_year','ann_sin','ann_cos']],
}
for name, fs in sets.items():
    print('%-20s 431: %.3f  403: %.3f' % (name, pseudo(fs,30,431), pseudo(fs,30,403)))
