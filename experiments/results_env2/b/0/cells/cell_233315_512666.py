import pandas as pd, numpy as np, agent_api
t = agent_api.load_saved('e011_pruned_basket.parquet')
drop = ['tenure','days_since_first','week','week_q',
        'spend_ly','ly_trail28','ly_future28',
        'total_spend','life_spend',
        'n_campaigns','camp_TypeA','camp_TypeB','camp_TypeC','camp_recent84',
        'camp_x_demo','campA_x_spend28',
        'redemp_total','days_since_redemp','z_active_wk_52']
drop = [c for c in drop if c in t.columns]
t2 = t.drop(columns=drop)
print('dropped', len(drop), '-> shape', t2.shape)
print('remaining cols:', len(t2.columns)-2)
p = agent_api.save_table(t2, 'e013_stationary.parquet')
print(p)
