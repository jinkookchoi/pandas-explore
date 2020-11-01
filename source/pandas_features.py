# -*- coding: utf-8 -*-

import os
import uuid
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def filter_by_tag(df, *tags, cond='and', return_cols=True):
    """1개 이상의 컬럼 키워드를 참조하여 and/or 조건으로 해당하는 컬럼 리스트를 반환한다.

    키워드 태그를 조건으로 검색하되, 해당하는 컬럼 리스트 혹은 Pandas 데이터프레임으로 반환가능하며,
    기본값으로 검색 조건은 'and', 컬럼 리스트 반환이 True 값이다.

    Arguments:
        df (dataframe): Pandas Data Frame
        tags : tuple of column string tags
        cond (string): filtering condition, "and" or "or"
        return_cols (boolean): Determine return type

    Returns:
        List of columns queried by tags

    Example:
        >>> # sample data frame
        >>> df = pd.DataFrame({
            'fruit_amt':np.random.random(2), 
            'juice_amt':np.random.random(2), 
            'sales_cnt':np.random.random(2)})

        >>> # filter by tag 'and' condition and get data frame
        >>> exp.filter_by_tag(df, 'amt', cond='and', return_cols=False)
         	fruit_amt	juice_amt
        0	0.239311	0.030956
        1	0.672927	0.938022

        >>> # filter by tag 'or' condition and get columns list
        >>> exp.filter_by_tag(df, 'fruit', 'sales', cond='or', return_cols=True)
            ['fruit_amt', 'sales_cnt']
    """
    col_intersectin_set = {}
    for idx, tag in enumerate(tags):
        selected_set = {col for col in df.columns.values if col.find(str(tag)) >= 0}
        if idx == 0:
            col_intersectin_set = selected_set
        else:
            if cond == 'and':
                col_intersectin_set &= selected_set
            elif cond == 'or':
                col_intersectin_set |= selected_set

    col_selected = list(col_intersectin_set)

    if return_cols:
        return col_selected
    else:
        if len(col_selected) > 0:
            return df.filter(list(col_intersectin_set))
        else:
            None

def csnap(df, fn=lambda x: x.shape, msg=None):
    """ Custom Help function to print things in method chaining.
        Returns back the df to further use in chaining.
    
    Example:
        >>> df.pipe(csnap, lambda x: x.sample(5))
    """
    if msg:
        print(msg)
    display(fn(df))
    return df


def setcols(df, fn=lambda x: x.columns.map('_'.join), cols=None):
    """Sets the column of the data frame to the passed column list.

    Example:
        >>> (
        >>>     iris.pipe(csnap, lambda x: x.head(), msg="Before")
        >>>     .pipe(
        >>>         setcols,
        >>>         fn=lambda x: x.columns.str.lower()
        >>>         .str.replace(r"\(cm\)", "")
        >>>         .str.strip()
        >>>         .str.replace(" ", "_"),
        >>>         cols=list("abcd")
        >>>     )
        >>>     .pipe(csnap, lambda x: x.head(), msg="After")
        >>> )
            sepal_length	sepal_width	petal_length	petal_width
        0	5.1	3.5	1.4	0.2
        1	4.9	3.0	1.4	0.2
        2	4.7	3.2	1.3	0.2
    """
    if cols:
        df.columns = cols
    else:
        df.columns = fn(df)
    return df

def cfilter(df, fn, axis="rows"):
    """ Custom Filters based on a condition and returns the df.
        function - a lambda function that returns a binary vector
        thats similar in shape to the dataframe
        axis = rows or columns to be filtered.
        A single level indexing

        >>> (
        >>>     iris.pipe(
        >>>         setcols,
        >>>         fn=lambda x: x.columns.str.lower()
        >>>         .str.replace(r"\(cm\)", "")
        >>>         .str.strip()
        >>>         .str.replace(" ", "_"),
        >>>     ).pipe(cfilter, lambda x: x.columns.str.contains("sepal"), axis="columns")
        >>> )

        	sepal_length	sepal_width
        0	5.1	3.5
        1	4.9	3.0
        2	4.7	3.2
    """
    if axis == "rows":
        return df[fn(df)]
    elif axis == "columns":
        return df.iloc[:, fn(df)]

def extractor(df, cols, pattern, new_cols=None):
    """컬럼의 텍스트에서 특정 패턴을 찾아 새로운 컬럼에 넣습니다.
    
    Example:
        >>> # City, State에서 도시명 추출하기. Chicago, IL -> Chicago
        >>> df = pd.DataFrame({'city': ['Chicago, IL']})
        >>> df.pipe(extractor, ['city'], "(.*), \w{2}", ['new_city'])
        	city
        0	Chicago
    """
    extracted = df[cols].apply(lambda x: x.str.extract(pattern, expand=False))
    
    df = df.copy()
    if new_cols:
        df[new_cols] = extracted
    else:
        df[cols] = extracted
    return df

def split(df, column, target_name, target_type, regex, del_column=True):
    """컬럼 데이터를 패턴을 기준으로 값 분리하여 반환한다.
    
    Example:
        >>> df = pd.DataFrame({'year_month': ['1980.06', '1976.11']})
        >>> df.pipe(split, column='year_month', target_name=['year', 'month'], target_type=[int, int], regex=r'(\d+)\.(\d+)')
        	year	month
        0	1980	6
        1	1976	11
    """
    df_rtn = df.copy()
    df_rtn[target_name] = df[column].astype(str).str.extract(regex)
    if del_column:
        df_rtn.drop(columns=column, inplace=True)
    
    return df_rtn.astype(dict(zip(list(target_name), list(target_type))))

def split_item(df, col, str_split, index, new_col=None):
    """컬럼 데이터를 패턴을 기준으로 분리한 후 특정 인덱스를 추출한다.

    Example:
        >>> df = pd.DataFrame({'year_month': ['1980.06', '1976.11']})
        >>> (df
        >>>     .pipe(split_item, col='year_month', str_split='.', index=0, new_col='year')
        >>>     .pipe(split_item, col='year_month', str_split='.', index=1, new_col='month'))
    """

    if new_col:
        df[new_col] = df[col].str.split(str_split).str[index]
    else:
        df[col] = df[col].str.split(str_split).str[index]
    return df

# df.assign(age = make_normalize(df.age))
make_normalize = lambda series_: (series_ - np.min(series_))/(np.max(series_) - np.min(series_))

# df.assign(fare = make_zscore(df.fare))
make_zscore = lambda series_: (series_ - np.mean(series_))/np.std(series_)

# get_col_name_ko('mob_trmn_cnt')
def get_col_name_ko(col, metafile='metadata.csv', df_meta=None, comb=True):
    if df_meta is None:
        df_meta = pd.read_csv(metafile)
    col_ko = df_meta.query(f'col_name == "{col}"').col_name_ko.values[0]
    if comb:
        return f'{col}({col_ko})'
    else:
        return col_ko