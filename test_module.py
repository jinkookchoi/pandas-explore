import numpy as np
import pandas as pd
import pandas_explore as exp
from IPython.display import display, HTML, Markdown, Latex, Image

df_stock = pd.read_csv('./data/stock.csv').assign(Date = lambda df: pd.to_datetime(df['Date'], infer_datetime_format=True))
df_diamonds = pd.read_csv('./data/diamonds.csv')
df_titanic = pd.read_csv('./data/titanic.csv').astype({'Survived':'object', 'Pclass':'object'})
df_billboard = pd.read_csv('./data/billboard.csv').drop(columns=['x76th.week', 'x66th.week', 'x67th.week', 'x68th.week', 'x69th.week', 'x70th.week', 'x71st.week',
                                                                   'x72nd.week', 'x73rd.week', 'x74th.week', 'x75th.week'])

# dataset_stat
def test_dataset_stat():
    assert set([type(v) for k, v in exp.dataset_stat(df_stock).items()]).issubset({float, int, np.int64, np.int32})
    assert set([type(v) for k, v in exp.dataset_stat(df_diamonds).items()]).issubset({float, int, np.int64, np.int32})
    assert set([type(v) for k, v in exp.dataset_stat(df_titanic).items()]).issubset({float, int, np.int64, np.int32})
    assert set([type(v) for k, v in exp.dataset_stat(df_billboard).items()]).issubset({float, int, np.int64, np.int32})

def test_get_freedman_bins():
    pass

