import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())
P = dict(n=1200,lr=0.03,md=7,mcw=40,ss=0.8,cs=0.7,rl=1.0)

def mk(n,lr,md,mcw,ss,cs,rl,obj):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective=obj,tree_method='hist',n_jobs=8,random_state=0)

t0=time.time()
oof_log = np.zeros(len(df)); oof_off = np.zeros(len(df)); seen=np.zeros(len(df))
for d in days:
    va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
    if len(tr)==0: continue
    # log target
    m = mk(P['n'],P['lr'],P['md'],P['mcw'],P['ss'],P['cs'],P['rl'],'reg:squarederror')
    m.fit(tr[feats], np.log1p(tr.future_spend_4w), eval_set=[(va[feats],np.log1p(va.future_spend_4w))], verbose=False)
    p_log = np.expm1(m.predict(va[feats]))
    # per-household residual offset from full-train model
    mfull = mk(P['n'],P['lr'],P['md'],P['mcw'],P['ss'],P['cs'],P['rl'],'reg:absoluteerror')
    mfull.fit(tr[feats],tr.future_spend_4w,verbose=False)
    r = tr.future_spend_4w - mfull.predict(tr[feats])
    hh = tr.assign(r=r).groupby('household_key').r.mean()
    for shrink in (0.5,0.7,1.0):
        p_off = mfull.predict(va[feats]) + shrink*va.household_key.map(hh).fillna(0).values
        idx = df.index.isin(va.index)
        if shrink==0.5: oof_off[idx]=p_off
    oof_log[df.index.isin(va.index)]=p_log
    seen[df.index.isin(va.index)]=1
    if d==days[-1]:
        pass
mask=seen>0
yv = df.future_spend_4w.values[mask]
print('log  OOF MAE', round(np.abs(oof_log[mask]-yv).mean(),3), 'R2', round(1-((oof_log[mask]-yv)**2).sum()/((yv-yv.mean())**2).sum(),4), round(time.time()-t0,1))
print('off  OOF MAE', round(np.abs(oof_off[mask]-yv).mean(),3), 'R2', round(1-((oof_off[mask]-yv)**2).sum()/((yv-yv.mean())**2).sum(),4))
