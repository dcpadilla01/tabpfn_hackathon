import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def mk(obj='reg:absoluteerror',n=1200,lr=0.03,md=7,mcw=10,ss=0.8,cs=0.7,rl=1.0):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective=obj,tree_method='hist',n_jobs=8,random_state=0)

def oof_preds(obj='reg:absoluteerror',**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(obj=obj,**kw)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    return oof, seen>0

def mae(p,mask):
    y=df.future_spend_4w.values[mask]; return np.abs(p[mask]-y).mean()

t0=time.time()
pA,m = oof_preds(n=2000,lr=0.02,md=7,mcw=10); print('lr.02', round(mae(pA,m),3), round(time.time()-t0,1))
pB,_ = oof_preds(obj='reg:squarederror',md=7,mcw=10); print('L2', round(mae(pB,m),3), round(time.time()-t0,1))
for w in (0.5,0.7):
    print('blend',w, round(mae(w*pA+(1-w)*pB,m),3))
