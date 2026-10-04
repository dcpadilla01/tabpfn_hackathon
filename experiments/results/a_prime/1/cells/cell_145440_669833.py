import pandas as pd, numpy as np, agent_api as A

tr_days = A.snapshot_days()['train']
t3 = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
base_feats = [c for c in t3.columns if c not in ('household_key','snapshot_day')]

def feats_for(tx, s):
    w = ((s - tx['day']) // 28).astype('int64')
    tx = tx.assign(w=w)
    g = tx.groupby(['household_key','w'])
    spend = g['sales_value'].sum().unstack()
    first_day = tx.groupby('household_key')['day'].min()
    elig = first_day.index[first_day <= s-84]
    spend = spend.reindex(index=elig, columns=list(range(13))).fillna(0.0)
    S = spend.values; n = len(elig)
    act = (S > 0).astype(float)
    f = pd.DataFrame(index=elig)
    # A: recency-decayed spend
    for hl, nm in [(2,'hl2'), (4,'hl4')]:
        wt = 0.5 ** (np.arange(6) / hl); wt = wt / wt.sum()
        f['dec_' + nm] = (S[:, :6] * wt).sum(1)
    # B: activity/level decomposition
    f['act_rate_3'] = act[:, :3].mean(1)
    f['act_rate_6'] = act[:, :6].mean(1)
    f['act_rate_13'] = act[:, :13].mean(1)
    act13 = act[:, :13].sum(1)
    mean_active_13 = np.where(act13 > 0, S[:, :13].sum(1) / np.maximum(act13, 1), 0.0)
    f['mean_active_13'] = mean_active_13
    f['expected_13'] = f['act_rate_13'] * mean_active_13
    a6 = act[:, :6].sum(1)
    f['expected_6'] = f['act_rate_6'] * np.where(a6 > 0, S[:, :6].sum(1) / np.maximum(a6, 1), 0.0)
    nwin = np.maximum(((s - first_day.loc[elig] + 1) // 28).values, 1)
    f['spend_per_win'] = S.sum(1) / nwin
    # C: dormancy gaps / resumption
    zrun = np.ones(n); zs = np.zeros(n)
    for k in range(13):
        zrun = zrun * (act[:, k] == 0); zs += zrun
    f['zero_streak'] = zs
    rs = np.zeros(n); rc = np.zeros(n)
    for k in range(1, 13):
        m = (act[:, k] == 0) & (act[:, k-1] > 0)
        rs[m] += S[m, k-1]; rc[m] += 1
    f['mean_resumption'] = np.where(rc > 0, rs / np.maximum(rc, 1), mean_active_13)
    # D: market seasonal anchor (target window last year = window w12)
    mkt_tot = S.sum(0); mkt_nact = act.sum(0)
    mkt_pa = np.where(mkt_nact > 0, mkt_tot / np.maximum(mkt_nact, 1), np.nan)
    f['mkt_tot_w12'] = mkt_tot[12]; f['mkt_nact_w12'] = mkt_nact[12]
    f['mkt_peract_w12'] = mkt_pa[12]; f['mkt_peract_w0'] = mkt_pa[0]
    r = mkt_pa[12] / mkt_pa[0] if mkt_pa[0] and mkt_pa[0] > 0 else np.nan
    f['mkt_ratio'] = r
    naive = S[:, :3].mean(1)
    f['naive_x_mkt'] = naive * r
    f['exp13_x_mkt'] = f['expected_13'] * r
    # E: household seasonal multiplier
    f['hh_seas_mult'] = S[:, 12] / np.maximum(f['spend_per_win'], 1e-9)
    # F: target-window annual phase
    ph = 2 * np.pi * ((s + 14) % 364) / 364
    f['sin_ann'] = np.sin(ph); f['cos_ann'] = np.cos(ph)
    return f

new = {}
for s in tr_days:
    v = A.snapshot(as_of_day=s)
    new[s] = feats_for(v.transactions, s).assign(snapshot_day=s)
newdf = pd.concat(new.values()).reset_index()
df = t3.merge(tt, on=['household_key','snapshot_day']).merge(newdf, on=['household_key','snapshot_day'], how='left')
print('merged', df.shape)

groups = {
 'A_recdecay': ['dec_hl2','dec_hl4'],
 'B_actlevel': ['act_rate_3','act_rate_6','act_rate_13','mean_active_13','expected_13','expected_6','spend_per_win'],
 'C_gaps': ['zero_streak','mean_resumption'],
 'D_mkt': ['mkt_tot_w12','mkt_nact_w12','mkt_peract_w12','mkt_peract_w0','mkt_ratio','naive_x_mkt','exp13_x_mkt'],
 'E_hhseas': ['hh_seas_mult'],
 'F_phase': ['sin_ann','cos_ann'],
}

def loso(cols, lams=(3,30,300)):
    Xall = df[cols].astype(float)
    mu = Xall.mean(); sd = Xall.std(); sd[sd==0]=1
    Xs = ((Xall - mu) / sd).fillna(0.0).values
    y = df['future_spend_4w'].values
    days = df['snapshot_day'].values
    best = (1e9, None)
    for lam in lams:
        errs = []
        for d in tr_days:
            m = days != d
            Xi, yi = Xs[m], y[m]
            A_ = np.hstack([np.ones((len(Xi),1)), Xi])
            R = A_.T @ A_ + lam*np.eye(Xi.shape[1]+1); R[0,0] -= lam
            w = np.linalg.solve(R, A_.T @ yi)
            Xo = np.hstack([np.ones((len(Xs[~m]),1)), Xs[~m]])
            errs.append(np.mean(np.abs(Xo @ w - y[~m])))
        e = float(np.mean(errs))
        if e < best[0]: best = (e, lam)
    return best

res = {}
base_only = loso(base_feats); res['base(E003)'] = base_only
print('base(E003):', round(base_only[0],2), 'lam', base_only[1])
acc = []
for gname, gcols in groups.items():
    acc += gcols
    r = loso(base_feats + acc)
    res['+'+gname] = r
    print(f'+{gname}: {round(r[0],2)} (lam {r[1]})  cum_feats={len(base_feats)+len(acc)}')
r = loso(base_feats + [c for g in groups.values() for c in g])
print('ALL:', round(r[0],2), 'lam', r[1])
# market-only and actlevel-only
print('+D only:', round(loso(base_feats + groups['D_mkt'])[0],2))
print('+B only:', round(loso(base_feats + groups['B_actlevel'])[0],2))