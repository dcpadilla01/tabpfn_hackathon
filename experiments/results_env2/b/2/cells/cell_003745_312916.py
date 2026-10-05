
import pandas as pd, numpy as np

NEWCOLS = ['x_spend_7d','x_spend_14d','x_trips_7d','x_trips_14d','x_lvl_ew2','x_f_ew_hl6','x_f_ew_hl8',
           'x_f56_mean','x_f84_mean','x_r_std','x_r_ew_hl8','x_zero_streak','x_active_frac13',
           'x_p_s28_rmean','x_p_s28_rmed','x_p_lvl_rmean','x_p_s28_fmean','x_p_usual_rmean','x_p_s28_carry','x_p_ya_rmean']

def fn(view, snapshot_day):
    s = int(snapshot_day)
    hh = pd.Index(np.asarray(view.households).ravel())
    tx = view.table('transactions')
    tx = tx[tx['household_key'].isin(hh)]
    ag = tx.groupby(['household_key','day']).agg(sp=('sales_value','sum'), nb=('basket_id','nunique')).reset_index()
    full = list(range(1, s+1))
    spm = ag.pivot(index='household_key', columns='day', values='sp').reindex(index=hh, columns=full).fillna(0.0)
    nbm = ag.pivot(index='household_key', columns='day', values='nb').reindex(index=hh, columns=full).fillna(0.0)
    Sv = spm.values; Nv = nbm.values; n = len(hh)
    Kmax = max(1, (s-28)//28 + 1)
    W = np.empty((n, Kmax))
    for k in range(Kmax):
        lo = max(0, s-28*(k+1)); hi = s-28*k
        W[:,k] = Sv[:, lo:hi].sum(1)
    s7 = Sv[:, max(0,s-7):s].sum(1); s14 = Sv[:, max(0,s-14):s].sum(1)
    t7 = Nv[:, max(0,s-7):s].sum(1); t14 = Nv[:, max(0,s-14):s].sum(1)
    fd = tx.groupby('household_key')['day'].min().reindex(hh).values.astype(float)
    Kh = np.clip(np.floor((s - fd - 27)/28).astype(int) + 1, 1, Kmax)
    out = {c: np.full(n, np.nan) for c in NEWCOLS}
    w2 = 0.5**(np.arange(Kmax)/2.0); w6 = 0.5**(np.arange(Kmax)/6.0); w8 = 0.5**(np.arange(Kmax)/8.0)
    for i in range(n):
        K = int(Kh[i]); seg = W[i,:K]; w0 = seg[0]
        lvl = (seg*w2[:K]).sum()/w2[:K].sum()
        f6 = (seg*w6[:K]).sum()/w6[:K].sum(); f8 = (seg*w8[:K]).sum()/w8[:K].sum()
        f56 = (seg[:-1]+seg[1:]).mean() if K>=2 else np.nan
        f84 = (seg[:-2]+seg[1:-1]+seg[2:]).mean() if K>=3 else np.nan
        r_mean=r_med=r_std=r_ew=np.nan
        m = seg[1:] > 0
        if m.any():
            r = seg[:-1][m]/seg[1:][m]
            r_mean=r.mean(); r_med=np.median(r); r_std=r.std()
            kk = np.arange(1,K)[m]; ww = 0.5**((kk-1)/8.0); r_ew=(r*ww).sum()/ww.sum()
        z=0
        while z<K and seg[z]==0: z+=1
        act = (seg[:min(13,K)]>0).mean()
        usual = np.median(seg[1:min(K,14)]) if K>=2 else np.nan
        ya = seg[13] if K>13 else np.nan
        carry = ((seg[:-1]>0)&(seg[1:]>0)).mean()
        out['x_spend_7d'][i]=s7[i]; out['x_spend_14d'][i]=s14[i]
        out['x_trips_7d'][i]=t7[i]; out['x_trips_14d'][i]=t14[i]
        out['x_lvl_ew2'][i]=lvl; out['x_f_ew_hl6'][i]=f6; out['x_f_ew_hl8'][i]=f8
        out['x_f56_mean'][i]=f56; out['x_f84_mean'][i]=f84
        out['x_r_std'][i]=r_std; out['x_r_ew_hl8'][i]=r_ew
        out['x_zero_streak'][i]=z; out['x_active_frac13'][i]=act
        out['x_p_s28_rmean'][i]=w0*r_mean; out['x_p_s28_rmed'][i]=w0*r_med
        out['x_p_lvl_rmean'][i]=lvl*r_mean; out['x_p_s28_fmean'][i]=w0*seg.mean()
        out['x_p_usual_rmean'][i]=usual*r_mean; out['x_p_s28_carry'][i]=w0*carry
        out['x_p_ya_rmean'][i]=ya*r_mean
    return pd.DataFrame(out, index=hh)

res = build_features(fn)
print('built', res.shape, sorted(res.snapshot_day.unique()))
print(res[NEWCOLS].isna().mean().round(3).to_dict())
e = load_saved('e011_table.parquet')
print('e011', e.shape)
comb = e.merge(res, on=['household_key','snapshot_day'], how='inner')
print('combined', comb.shape)
assert len(comb)==36426, len(comb)
assert comb.duplicated(['household_key','snapshot_day']).sum()==0
p = save_table(comb, 'e019_table')
print('saved', p)
