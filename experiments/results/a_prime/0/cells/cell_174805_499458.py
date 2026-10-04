
import numpy as np, pandas as pd

DEPTS = ['DELI','DRUG GM','GROCERY','KIOSK-GAS','MEAT','MEAT-PCKGD','MISC SALES TRAN','NUTRITION','PASTRY','PRODUCE']

def fn(view, snap):
    try:
        hh_src = view.households['household_key']
    except Exception:
        hh_src = view.households
    hh_idx = pd.Index(hh_src)
    t = view.transactions
    t = t[t['household_key'].isin(hh_idx)][['household_key','basket_id','day','sales_value','product_id']]
    n = len(hh_idx)
    g = t.groupby('household_key')
    first_day = g['day'].min().reindex(hh_idx)
    last_day = g['day'].max().reindex(hh_idx)
    rec = (snap - last_day).fillna(999).clip(0, 336).astype(float)
    off = snap - t['day']
    out = pd.DataFrame(index=hh_idx)

    blk = (off // 28).astype(int)
    bs = t.assign(blk=blk).groupby(['household_key','blk'])['sales_value'].sum()
    mat = bs.unstack(fill_value=0.0).reindex(columns=range(12), fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    mv = mat.values
    zmat = mv <= 0.0
    tb = np.floor((snap - first_day.fillna(snap)) / 28.0)
    tb = np.clip(tb.astype(int).values + 1, 1, 12)
    lead = np.zeros(n); run = np.zeros(n); frac = np.zeros(n)
    for i in range(n):
        z = zmat[i, :tb[i]]
        lead[i] = int(np.argmax(~z)) if (~z).any() else tb[i]
        best = cur = 0
        for v in z:
            cur = cur + 1 if v else 0
            if cur > best: best = cur
        run[i] = best
        frac[i] = z.mean()
    out['zb_consec'] = lead
    out['zb_run_max'] = run
    out['zb_frac'] = frac
    out['zb_zero84'] = (mv[:, :3].sum(1) <= 0).astype(float)

    def win(days):
        w = t[off < days]
        s = w.groupby('household_key')['sales_value'].sum().reindex(hh_idx, fill_value=0.0)
        b = w.groupby('household_key')['basket_id'].nunique().reindex(hh_idx, fill_value=0)
        d = w.groupby('household_key')['day'].nunique().reindex(hh_idx, fill_value=0)
        return s, b, d
    s7, b7, d7 = win(7)
    s14, b14, d14 = win(14)
    s28, b28, d28 = win(28)
    s84, b84, _ = win(84)
    out['b7'] = b7.values; out['b14'] = b14.values
    out['s7'] = s7.values; out['s14'] = s14.values
    out['days14'] = d14.values
    out['spd_day28'] = np.clip(s28.values / (d28.values + 1.0), 0, 1000)
    out['bvt28_84'] = np.clip((s28.values / (b28.values + 1.0)) /
                              ((s84.values / (b84.values + 1.0)) + 1.0), 0, 5)

    dd = t[['household_key','day']].drop_duplicates().sort_values(['household_key','day'])
    dd['gp'] = dd.groupby('household_key')['day'].diff()
    gm = dd.groupby('household_key')['gp'].mean().reindex(hh_idx).fillna(28).clip(0, 180)
    gmd = dd.groupby('household_key')['gp'].median().reindex(hh_idx).fillna(28).clip(0, 180)
    out['gap_mean'] = gm.values; out['gap_med'] = gmd.values
    out['ovd_ratio'] = np.clip(rec.values / (gm.values + 1.0), 0, 10)
    out['ovd_mratio'] = np.clip(rec.values / (gmd.values + 1.0), 0, 10)
    out['ovd_flag'] = (rec.values > 1.25 * gm.values + 2).astype(float)

    t84 = t[off < 84]
    try:
        pr = view.products[['product_id','department']]
        t84 = t84.merge(pr, on='product_id', how='left')
        dep = t84['department'].astype(str)
    except Exception:
        dep = pd.Series(['other'] * len(t84), index=t84.index)
    dep = np.where(dep.isin(DEPTS), dep, 'other')
    pv = t84.assign(dep=dep).pivot_table(index='household_key', columns='dep',
                                         values='sales_value', aggfunc='sum')
    pv = pv.reindex(columns=DEPTS + ['other'], fill_value=0.0).reindex(hh_idx, fill_value=0.0)
    shares = pv.div(pv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).values
    s84v = s84.reindex(hh_idx).fillna(0).values
    b84v = b84.reindex(hh_idx).fillna(0).values
    F = np.column_stack([shares, np.log1p(s84v) / 7.0, np.log1p(b84v) / 4.0, rec.values / 84.0])
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    nr = np.linalg.norm(Fz, axis=1); nr[nr == 0] = 1.0
    Fn = Fz / nr[:, None]
    S = Fn @ Fn.T
    np.fill_diagonal(S, -1.0)
    k = 25
    idxp = np.argpartition(-S, k, axis=1)[:, :k]
    sims = np.take_along_axis(S, idxp, axis=1)
    w = np.clip(sims, 0, None)
    ws = w.sum(1) + 1e-9
    s28v = s28.reindex(hh_idx).fillna(0).values
    b28v = b28.reindex(hh_idx).fillna(0).values.astype(float)
    act = (b28v > 0).astype(float)
    sN = s28v[idxp]; bN = b28v[idxp]; aN = act[idxp]
    out['kn2_sim'] = w.sum(1) / k
    out['kn2_s28'] = (w * sN).sum(1) / ws
    out['kn2_b28'] = (w * bN).sum(1) / ws
    out['kn2_act'] = (w * aN).sum(1) / ws
    out['kn2_ratio'] = np.clip(((w * sN).sum(1) / ws) / (s28v + 1.0), 0, 10)
    return out

new = agent_api.build_features(fn)
print('new block:', new.shape, 'snapshots:', sorted(new.snapshot_day.unique()))
base = agent_api.load_saved('e019_knn.parquet')
m = base.merge(new, on=['household_key','snapshot_day'], how='left')
print('merged:', m.shape)
cols = [c for c in new.columns if c not in ('household_key','snapshot_day')]
print('NaNs:', m[cols].isna().sum().sum())
tt = agent_api.train_targets()
tr = m.merge(tt, on=['household_key','snapshot_day'])
for c in cols:
    print(f'{c:12s} corr={tr[c].corr(tr.future_spend_4w): .4f}')
path = agent_api.save_table(m, 'e020_final')
print('saved:', path)
