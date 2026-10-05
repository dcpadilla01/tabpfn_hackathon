import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def fit_pred(tr, va, params, feats, target='future_spend_4w', objective='reg:absoluteerror'):
    m = xgb.XGBRegressor(n_estimators=params['n'], learning_rate=params['lr'],
        max_depth=params['md'], min_child_weight=params['mcw'], subsample=params['ss'],
        colsample_bytree=params['cs'], reg_lambda=params['rl'], gamma=params.get('g',0.0),
        objective=objective, tree_method='hist', n_jobs=8, random_state=0)
    y = tr[target]
    if objective=='reg:absoluteerror': ytr = y
    else: ytr = y
    m.fit(tr[feats], ytr, eval_set=[(va[feats], va[target])], verbose=False)
    return m.predict(va[feats])

t0=time.time()
def run_oof(mode, params=dict(n=1200,lr=0.03,md=7,mcw=40,ss=0.8,cs=0.7,rl=1.0)):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        if mode=='base':
            p = fit_pred(tr,va,params,feats)
        elif mode=='log':
            tr2 = tr.copy(); tr2['y']=np.log1p(tr2.future_spend_4w)
            p = np.expm1(fit_pred(tr2,va,params,feats,target='y'))
        elif mode=='hh_offset':
            p = fit_pred(tr,va,params,feats)
            # per-household mean residual from train
            tr2 = tr.copy(); tr2['res']=tr2.future_spend_4w-p  # careful: p is for va rows
            # need residuals on train: refit quick? instead compute via crossval within train
            # simpler: use per-household mean of (target - pred) where pred from a model trained on other train days
            # approximate: residual = target - model trained on all train
            mfull = xgb.XGBRegressor(n_estimators=params['n'],learning_rate=params['lr'],max_depth=params['md'],
                min_child_weight=params['mcw'],subsample=params['ss'],colsample_bytree=params['cs'],
                reg_lambda=params['rl'],objective='reg:absoluteerror',tree_method='hist',n_jobs=8,random_state=0)
            mfull.fit(tr[feats],tr.future_spend_4w,verbose=False)
            r = tr.future_spend_4w - mfull.predict(tr[feats])
            hh = tr.assign(r=r).groupby('household_key').r.mean()
            shrunken = hh*0.7
            p = p + va.household_key.map(shrunken).fillna(0).values
        oof[df.index.isin(va.index)]=p; seen[df.index.isin(va.index)]=1
    m_ = seen>0
    mae = np.abs(oof[m_]-df.future_spend_4w.values[m_]).mean()
    print(mode,'OOF MAE',round(mae,3),'time',round(time.time()-t0,1)); return mae

run_oof('base')
run_oof('log')
run_oof('hh_offset')
