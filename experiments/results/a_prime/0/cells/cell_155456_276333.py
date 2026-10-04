
import agent_api, pandas as pd, numpy as np

def make_fn():
    def fn(view, sday):
        tx = view.transactions
        g = tx.groupby('household_key').day.max()
        hh_all = g.index
        last = g.values.astype(float)
        K = 16
        BS = {}; BB = {}
        for k in range(K+1):
            d1, d2 = sday-28*k-28, sday-28*k
            m = (tx.day > d1) & (tx.day <= d2)
            sub = tx[m]
            BS[k] = sub.groupby('household_key').sales_value.sum().reindex(hh_all).fillna(0).values
            BB[k] = sub.groupby('household_key').basket_id.nunique().reindex(hh_all).fillna(0).values
        REC = {k: np.clip(sday-28*k - last, 0, None) for k in range(K)}
        def design(k):
            s1,s2,s3,s4 = BS[k+1],BS[k+2],BS[k+3],BS[k+4]
            return np.column_stack([s1,s2,s3,s4,BB[k+1],REC[k],s1/(s1+s2+s3+1),(s1>0).astype(float)])
        def design_sp(k):
            s1,s2,s3,s4 = BS[k+1],BS[k+2],BS[k+3],BS[k+4]
            return np.column_stack([s1,s2,s3,s4])
        def ridge_fit(X, y, w, alpha):
            Xw = X*np.sqrt(w)[:,None]
            A = np.hstack([np.ones((len(X),1)), Xw])
            return np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@(y*np.sqrt(w)))
        X0 = design(0); X0sp = design_sp(0)
        out = pd.DataFrame(index=hh_all)
        for Kfit, tag in [(9,'9'),(6,'6'),(12,'12')]:
            Xs=[];ys=[];ws=[]
            for k in range(1,Kfit+1):
                Xs.append(design(k)); ys.append(BS[k]); ws.append(np.full(len(hh_all), 0.9**(k-1)))
            wtr = ridge_fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), 1.0)
            out['emb_pred_'+tag] = np.hstack([np.ones((len(X0),1)), X0]) @ wtr
        Xs=[];ys=[];ws=[]
        for k in range(1,10):
            Xs.append(design_sp(k)); ys.append(BS[k]); ws.append(np.full(len(hh_all), 0.9**(k-1)))
        wtr = ridge_fit(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), 1.0)
        out['emb_sp'] = np.hstack([np.ones((len(X0sp),1)), X0sp]) @ wtr
        out['emb_clip'] = np.clip(out['emb_pred_9'].values, 0, None)
        out['emb_ratio'] = out['emb_pred_9'].values/(BS[1]+1)
        out['emb_diff'] = out['emb_pred_9'].values - BS[1]
        need = pd.Index(view.households) if not hasattr(view.households, 'columns') else pd.Index(view.households.household_key)
        out = out.reindex(need)
        for c in out.columns: out[c] = out[c].fillna(0)
        return out
    return fn

F = agent_api.build_features(make_fn())
print('F:', F.shape, list(F.columns)[:12])
print(F.head(3))
base = agent_api.load_saved('e011_rank.parquet')
print('base:', base.shape)
if 'snapshot_day' in F.columns:
    M = base.merge(F, on=['household_key','snapshot_day'], how='left')
else:
    F2 = F.reset_index().rename(columns={'index':'household_key'})
    M = base.merge(F2, on=['household_key','snapshot_day'], how='left')
print('merged:', M.shape, 'NaN emb:', M[[c for c in M.columns if c.startswith('emb')]].isna().mean().round(3).to_dict())
path = agent_api.save_table(M, 'e012_embed.parquet')
print('saved:', path)
