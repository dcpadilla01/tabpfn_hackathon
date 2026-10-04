import agent_api as A
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
merged = A.load_saved('e006_temporal.parquet')
mkt = A.load_saved('mkt_v2.parquet')
tt = A.train_targets()
newf = [c for c in merged.columns if c not in mkt.columns and c not in ('household_key','snapshot_day')]
mktf = [c for c in mkt.columns if c not in ('household_key','snapshot_day')]

def screen(df, feats):
    dd = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    itr = dd['snapshot_day'].values <= 403
    X = dd[feats].apply(pd.to_numeric, errors='coerce').values.astype(float)
    yy = dd['future_spend_4w'].values
    med = np.nanmedian(X[itr], 0)
    X = np.where(np.isnan(X), med, X)
    Xtr, ytr = X[itr], yy[itr]; Xva, yva = X[~itr], yy[~itr]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-9] = 1
    Ztr = np.c_[(Xtr-mu)/sd, np.ones(len(Xtr))]; Zva = np.c_[(Xva-mu)/sd, np.ones(len(Xva))]
    res = []
    for use_log in (False, True):
        ty = np.log1p(ytr) if use_log else ytr
        for a in (3, 30, 300, 3000):
            Am = Ztr.T@Ztr + a*np.eye(Ztr.shape[1]); Am[-1,-1] -= a
            w = np.linalg.solve(Am, Ztr.T@ty)
            pv = Zva@w
            if use_log: pv = np.expm1(np.clip(pv, 0, 12))
            res.append(('log' if use_log else 'raw', a, np.abs(pv-yva).mean()))
    return min(res, key=lambda r: r[2])

print('full fixed block:', screen(merged, mktf + newf))
curated = ['slope','p12','yoy_diff','seas_ratio','nb12','p5','w5','w2','p6','w_slope','b_mean','nb3','p11','nb11','p3','w8','nb5','w7','nb4','p1','n_blocks_full','w3','nb_std','nb_max','w_cv','nb2','p2','nb1','w1','p1_div_bmean']
print('curated 30:', screen(merged, mktf + curated))
print('mkt base:', screen(merged, mktf))