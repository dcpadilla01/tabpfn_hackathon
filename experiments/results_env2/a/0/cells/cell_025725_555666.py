
import agent_api as A, numpy as np, pandas as pd, time, xgboost as xgb
t0=time.time()
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); cand = A.load_saved('cand_train.parquet')
train_days = sorted(tt.snapshot_day.unique())
key = ['household_key','snapshot_day']
fcols = [c for c in feats.columns if c not in key]
demo_cols = [c for c in fcols if feats[c].dtype==object]
print('string cols:', demo_cols)
def enc(df):
    df = df.copy()
    for c in demo_cols: df[c] = pd.Categorical(df[c].astype(str)).codes
    return df
feats = enc(feats)
candc = ['units_84','spend84','trips_7','nprod_28','ncomm_28','up_mean_84','premium_idx','store_rich_84','disp_share_84','mail_share_84']
tr = feats[feats.snapshot_day.isin(train_days)].merge(tt, on=key).merge(cand[key+candc], on=key, how='left')
print('tr', tr.shape, 'cand missing:', tr[candc[0]].isna().sum())
for c in ['units_84','spend84','trips_7','nprod_28','ncomm_28','disp_share_84','mail_share_84','up_mean_84']: tr[c] = tr[c].fillna(0.0)
tr['premium_idx'] = tr.premium_idx.fillna(1.0)
tr['store_rich_84'] = tr.store_rich_84.fillna(tr.store_rich_84.median())
y = tr.future_spend_4w.values
Xb = tr[fcols].values.astype(np.float32)
Xn = tr[fcols+candc].values.astype(np.float32)
print('Xb', Xb.shape, 'Xn', Xn.shape)
CONFIGS = [dict(max_depth=4,min_child_weight=20), dict(max_depth=5,min_child_weight=40),
           dict(max_depth=4,min_child_weight=40), dict(max_depth=6,min_child_weight=60)]
def bag(Xtr, ytr, Xte):
    p = np.zeros(len(Xte))
    for c in CONFIGS:
        prm = {'objective':'reg:quantileerror','quantile_alpha':0.5,'learning_rate':0.08,
               'n_estimators':400,'nthread':4,'seed':0, 'base_score':float(np.median(ytr))}
        prm.update(c)
        m = xgb.XGBRegressor(**prm); m.fit(Xtr, ytr)
        p += m.predict(Xte)
    return p/len(CONFIGS)
days = tr.snapshot_day.values
def loso(X):
    oof = np.zeros(len(tr))
    for d in train_days:
        te = days==d
        oof[te] = bag(X[~te], y[~te], X[te])
    return oof
t1=time.time(); oof_b = loso(Xb); print('arm a (base) LOSO MAE %.4f  [%.0fs]' % (np.abs(oof_b-y).mean(), time.time()-t1))
t1=time.time(); oof_n = loso(Xn); print('arm b (+cand) LOSO MAE %.4f  [%.0fs]' % (np.abs(oof_n-y).mean(), time.time()-t1))
med = np.median(y)
for gname in ['wk_cv8','wk_std8']:
    gvar = tr[gname].values
    q = np.quantile(gvar, [1/3, 2/3]); grp = np.digitize(gvar, q)
    pred = oof_b.copy()
    for te_day in train_days:
        te = days==te_day
        for g in range(3):
            m_te = te & (grp==g); m_tr = (~te) & (grp==g)
            if m_tr.sum()<100: continue
            ss = np.linspace(0.5,1.0,11)
            maes = [np.abs(med + s*(oof_b[m_tr]-med) - y[m_tr]).mean() for s in ss]
            s_star = ss[int(np.argmin(maes))]
            pred[m_te] = med + s_star*(oof_b[m_te]-med)
    print('gated shrink by %s: CV MAE %.4f' % (gname, np.abs(pred-y).mean()))
print('total %.0fs' % (time.time()-t0))
