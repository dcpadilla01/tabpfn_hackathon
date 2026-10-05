
import agent_api, pandas as pd, numpy as np, time

def fn(view, snap):
    hh = view.households
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(set(hh))][['household_key','basket_id','day','sales_value','product_id']]
    prod = view.table('products')[['product_id','department']]
    tx = tx.merge(prod, on='product_id', how='left')
    g = tx.groupby(['household_key','basket_id'], as_index=False).agg(day=('day','first'), spend=('sales_value','sum'))
    out = pd.DataFrame(index=hh)
    # disjoint 28d windows s1..s4
    for i,(lo,hi) in enumerate([(snap-27,snap),(snap-55,snap-28),(snap-83,snap-56),(snap-111,snap-84)]):
        m = g[(g.day>=lo)&(g.day<=hi)].groupby('household_key').spend.sum()
        out[f's{i+1}'] = m.reindex(hh).fillna(0.0)
    s = out[['s1','s2','s3','s4']].values
    out['seq_mean'] = s.mean(1); out['seq_std'] = s.std(1)
    out['seq_cv'] = out['seq_std']/(out['seq_mean']+1)
    out['ratio_s1_s3'] = out.s1/(out.s3+1)
    out['ratio_s1_s4'] = out.s1/(out.s4+1)
    # dept-level: spend last 28d, days since last dept purchase
    top = ['GROCER','PRODUC','MEAT','DRUG G','DELI']
    t28 = tx[(tx.day>snap-28)&(tx.department.isin(top))]
    out['dept28_tot'] = t28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
    for d in top:
        td = tx[tx.department==d]
        out[f'dsp28_{d[:4]}'] = td[td.day>snap-28].groupby('household_key').sales_value.sum().reindex(hh).fillna(0.0)
        last = td.groupby('household_key').day.max()
        out[f'drec_{d[:4]}'] = (snap-last).reindex(hh).fillna(999.0)
    # basket gaps in last 112d
    b = g[(g.day>snap-112)].sort_values(['household_key','day'])
    b['gap'] = b.groupby('household_key').day.diff()
    gg = b.groupby('household_key').gap
    out['gap_mean'] = gg.mean().reindex(hh)
    out['gap_std'] = gg.std().reindex(hh)
    out['gap_max'] = gg.max().reindex(hh)
    out['gap_nbig'] = gg.apply(lambda x:(x>21).sum()).reindex(hh).fillna(0.0)
    lastb = b.groupby('household_key').day.max()
    out['last_gap'] = (snap-lastb).reindex(hh)
    # weekly activity streaks, last 12 weeks
    wk = (snap+8)//7
    txb = g.copy(); txb['w'] = (txb.day+8)//7
    aw = txb[(txb.w>wk-12)&(txb.w<=wk)].groupby(['household_key','w']).size().unstack(fill_value=0)
    aw = aw.reindex(hh).fillna(0.0)
    A = (aw>0).values.astype(int)
    cur_inact = np.zeros(len(hh)); streak=0; best=0
    for j in range(A.shape[1]):
        col = A[:,j]
        cur_inact = np.where(col==0, cur_inact+1, 0)
        streak = np.where(col==1, streak+1, 0)
        best = np.maximum(best, streak)
    out['wk_inact_streak'] = cur_inact; out['wk_best_streak'] = best
    out['spend28_vs_84'] = out.s1/(out.s1*0+1)  # placeholder replaced below
    return out.drop(columns=['spend28_vs_84'])

t0=time.time()
NF = agent_api.build_features(fn)
print('new feats', NF.shape, f'{time.time()-t0:.0f}s')
print(NF.head(3).to_string())
agent_api.save_table(NF, 'e005_newfeats.parquet')
