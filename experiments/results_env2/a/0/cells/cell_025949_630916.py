
import agent_api as A, numpy as np, pandas as pd, time, xgboost as xgb
t0=time.time()
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); oof = A.load_saved('oof_e013.parquet')
cand = A.load_saved('cand_train.parquet')
key = ['household_key','snapshot_day']
tr = tt.merge(oof, on=key).merge(cand, on=key, how='left').merge(feats[key+['wk_cv8','wk_std8','spend_28','exp4w_blend']], on=key, how='left')
print('tr', tr.shape, 'cand missing:', tr.units_84.isna().sum())
for c in ['units_84','spend84','trips_7','nprod_28','ncomm_28','disp_share_84','mail_share_84','up_mean_84']: tr[c] = tr[c].fillna(0.0)
tr['premium_idx'] = tr.premium_idx.fillna(1.0); tr['store_rich_84'] = tr.store_rich_84.fillna(tr.store_rich_84.median())
y = tr.future_spend_4w.values; base = tr.oof.values; r = y - base
days = tr.snapshot_day.values; tdays = sorted(tt.snapshot_day.unique())
print('resid: mean %.2f median %.2f std %.1f' % (r.mean(), np.median(r), r.std()))
def mae(p): return np.abs(p-y).mean()
print('base OOF MAE %.4f' % mae(base))
# 1) residual model g(cand), honest LOSO
candc = ['units_84','spend84','trips_7','nprod_28','ncomm_28','up_mean_84','premium_idx','store_rich_84','disp_share_84','mail_share_84']
Xc = tr[candc].values.astype(np.float32)
oof_g = np.zeros(len(tr))
for d in tdays:
    te = days==d
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=3, min_child_weight=50,
                         learning_rate=0.1, n_estimators=150, nthread=4, seed=0)
    m.fit(Xc[~te], r[~te]); oof_g[te] = m.predict(Xc[te])
print('g std on oof %.3f' % oof_g.std())
for w in [0.25,0.5,1.0]:
    print('base + %.2f*g(cand): LOSO MAE %.4f' % (w, mae(base + w*oof_g)))
# 2) constant shift
print('base+median(r): %.4f | base+0.5*mean(r): %.4f' % (mae(base+np.median(r)), mae(base+0.5*r.mean())))
# 3) clip tests
for c in [300,400,500,600,800,1000]:
    print('clip@%d: %.4f' % (c, mae(np.minimum(base,c))))
for q in [0.98,0.99,0.995]:
    c = np.quantile(base,q); print('clip@q%.3f(=%.0f): %.4f' % (q,c,mae(np.minimum(base,c))))
# 4) gated shrink by wk_cv8 terciles (honest over days)
for gname in ['wk_cv8','wk_std8']:
    gvar = tr[gname].values; q3 = np.nanquantile(gvar,[1/3,2/3]); grp = np.digitize(gvar,q3)
    pred = base.copy()
    for d in tdays:
        te = days==d
        for g in range(3):
            mte = te&(grp==g); mtr = (~te)&(grp==g)
            if mtr.sum()<100 or mte.sum()==0: continue
            ss = np.linspace(0.6,1.0,9)
            s = ss[int(np.argmin([np.abs(np.median(r[mtr]) + s*(base[mtr]-np.median(r[mtr])) - y[mtr]).mean() for s in ss]))]
            pred[mte] = np.median(r[mtr]) + s*(base[mte]-np.median(r[mtr]))
    print('gated shrink by %s: %.4f' % (gname, mae(pred)))
print('total %.0fs' % (time.time()-t0))
