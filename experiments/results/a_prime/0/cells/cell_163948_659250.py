import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

DEPTS=['GROCERY','DRUG GM','MEAT','PRODUCE','MEAT-PCKGD','DELI','KIOSK-GAS','PASTRY']

def fn(view, snapshot_day):
    d = snapshot_day
    hh = pd.Index(view.households)
    try:
        b = agent_api.load_saved("e013_stock.parquet")
        sub = b[b.snapshot_day==d].set_index('household_key')
        X = sub.drop(columns=['snapshot_day']).reindex(hh)
        def hg(col, edges, pfx, fill=0.0):
            x = X[col].fillna(fill).clip(lower=0).astype(float).values
            for e in edges: X[f"{pfx}h{int(e)}"] = np.maximum(0.0, x-e)
        hg('x_exp4w',[10,25,50,100,200,400,800],'hx_')
        hg('spend_84',[25,100,300],'hs84_')
        hg('x_spend_p1',[10,25,50,100,200,400,800],'hp1_')
        hg('x_spend_p2',[10,25,50,100,200,400,800],'hp2_')
        hg('recency',[7,14,28,56,112],'hr_')
        hg('x_spend_12',[25,100,300],'hs12_')
        hg('baskets_28',[1,3,6,10],'hb28_')
        hg('x_bv_4',[10,25,50,100],'hbv_')
        hg('days_28',[1,4,8,14],'hd28_')
        x = X['x_exp4w'].fillna(0).clip(lower=0).values.astype(float)
        bns = np.digitize(x, [0,10,25,50,75,100,150,200,300,450,600])
        for k in range(12): X[f'ohx_{k}'] = (bns==k).astype(float)
        t = view.transactions[['household_key','day','sales_value','product_id']].merge(
            view.products[['product_id','department']].astype({'department':'str'}), on='product_id', how='left')
        t['department']=t['department'].fillna('other').astype(str)
        t['dept8']=np.where(t.department.isin(DEPTS), t.department, 'other')
        w28=t[(t.day>=d-27)&(t.day<=d)]; w84=t[(t.day>=d-83)&(t.day<=d)]
        s28=w28.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
        s84=w84.groupby(['household_key','dept8']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
        for c in DEPTS+['other']:
            X[f'd28_{c}'] = s28[c].values if c in s28.columns else 0.0
            if c in s28.columns and c in s84.columns:
                X[f'r2884_{c}'] = (s28[c]/s84[c].replace(0,np.nan)).fillna(1.0).values
            else:
                X[f'r2884_{c}'] = 1.0
        CONSUM={'GROCERY','MEAT','PRODUCE','DELI','MEAT-PCKGD','SEAFOOD','SEAFOOD-PCKGD','PASTRY','DAIRY','FROZEN','BAKERY','KIOSK-GAS'}
        t['is_cons']=t.department.isin(CONSUM).astype(float)
        c28=w28.groupby('household_key').is_cons.sum().reindex(hh).fillna(0.0)
        a28=w28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        X['cons_share28']=(c28/a28.replace(0,np.nan)).fillna(0).values
        c84=w84.groupby('household_key').is_cons.sum().reindex(hh).fillna(0.0)
        a84=w84.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        X['cons_share84']=(c84/a84.replace(0,np.nan)).fillna(0).values
        lastc=w28[w28.is_cons==1].groupby('household_key').day.max()
        X['days_since_cons']=(d-lastc).reindex(hh).fillna(28).astype(float).values
        return X
    except Exception as e:
        print("FN-ERR", d, repr(e)[:200])
        F=pd.DataFrame(index=hh)
        tr=view.transactions
        F['spend_28']=tr[(tr.day>=d-27)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        F['spend_84']=tr[(tr.day>=d-83)].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        F['baskets_28']=tr[(tr.day>=d-27)].groupby('household_key').basket_id.nunique().reindex(hh).fillna(0.0)
        F['recency']=(d-tr.groupby('household_key').day.max()).reindex(hh).astype(float)
        F['tenure']=(d-tr.groupby('household_key').day.min()).reindex(hh).astype(float)
        return F

out = agent_api.build_features(fn)
print("built:", out.shape)
newcols=[c for c in out.columns if c.startswith(('hx_','hs84_','hp1_','hp2_','hr_','hs12_','hb28_','hbv_','hd28_','ohx_','d28_','r2884_','cons_','days_since'))]
print("n new cols:", len(newcols), "| sample:", newcols[:6], newcols[-4:])
print("NaN frac in new cols: %.4f" % np.mean([out[c].isna().mean() for c in newcols]))
print("snapshots:", sorted(out.snapshot_day.unique()))
p = agent_api.save_table(out, "e016_flex.parquet")
print("saved:", p)
