
import pandas as pd
def probe(view, sd):
    f = pd.DataFrame({'x': 1.0}, index=pd.Index(view.households, name='household_key'))
    return f
out = build_features(probe)
print(type(out), out.shape)
print(out.columns.tolist())
print(out.head(3))
print('index name:', out.index.name)
