import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def mk(n,lr,md,mcw,ss,cs,rl):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective='reg:absoluteerror',tree_method='hist',n_jobs=8,random_state=0)

def oof_mae(params):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(**params)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    mask=seen>0
    return np.abs(oof[mask]-df.future_spend_4w.values[mask]).mean(), oof, mask

t0=time.time()
configs = {
 'md5':  dict(n=1500,lr=0.03,md=5,mcw=40,ss=0.8,cs=0.7,rl=1.0),
 'md9':  dict(n=1200,lr=0.03,md=9,mcw=40,ss=0.8,cs=0.7,rl=1.0),
 'mcw10':dict(n=1200,lr=0.03,md=7,mcw=10,ss=0.8,cs=0.7,rl=1.0),
 'mcw100':dict(n=1200,lr=0.03,md=7,mcw=100,ss=0.8,cs=0.7,rl=1.0),
}
res={}
for k,p in configs.items():
    mae,oof,mask = oof_mae(p)
    res[k]=(mae,oof,mask)
    print(k, round(mae,3), round(time.time()-t0,1))
