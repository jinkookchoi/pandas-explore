# -*- coding: utf-8 -*-

import os
import uuid
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import markdown2
import tabulate
from IPython.display import display, HTML, Markdown, Latex, Image

"""pandas_explore.py: Pandasd Profiling Tools"""

__author__ = "Jin-kook Choi"
__license__ = "GPL"
__version__ = "0.0.1"
__email__ = "jinkookchoi@gmail.com"
__status__ = "Development"

def filter_by_tag(df, *tags, cond='and', return_cols=True):
    """Returns the column corresponding to and/or condition based on the tag list

    Search columns of data frame  based on tags and search conditions, or return filtered data frames

    Args:
        df (dataframe): Pandas Data Frame
        tags : tuple of column string tags
        cond (string): filtering condition, "and" or "or"
        return_cols (boolean): Determine return type

    Returns:
        List of columns queried by tags

    Example:
        >>> df = pd.DataFrame({
            'fruit_amt':np.random.random(2), 
            'juice_amt':np.random.random(2), 
            'sales_cnt':np.random.random(2)})
        >>> exp.filter_by_tag(df, 'amt', cond='and', return_cols=False)
        	fruit_amt	juice_amt
        0	0.239311	0.030956
        1	0.672927	0.938022
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

def get_dataset_stat(df):
    """Extract statistics from data frame

    Args:
        df (dataframe)
    
    Example:
        >>> exp.get_dataset_stat(df_stock)
        {'Number of variables': 7,
        'Number of observations': 5031,
        'Missing cells': 0,
        'Missing cells(%)': 0.0,
        'Duplicated rows': 0,
        'Duplicated rows(%)': 0.0}
    """
    if df.empty:
        return None
    else:
        num_of_obs, num_of_var = df.shape
        num_of_missing_cells = df.isna().sum().sum()
        
        f2 = lambda x: float(int(x * 1000))/1000
        
        ratio_of_missing_cells = f2(num_of_missing_cells / (num_of_obs * num_of_var) * 100)
        num_of_duplicated = df.duplicated().sum()
        ratio_of_duplicated = f2(num_of_duplicated / num_of_obs * 100)

        return {
            'Number of variables': num_of_var, 'Number of observations': num_of_obs,
            'Missing cells': num_of_missing_cells, 'Missing cells(%)': ratio_of_missing_cells,
            'Duplicated rows': num_of_duplicated, 'Duplicated rows(%)': ratio_of_duplicated
        }

def get_image_folder(image_fn=None, image_dir='image'):
    """Returns plot graphic directory

    When image_fn is not None, concatenate image folder and file name

    Args:
        image_fn (string): Image file name, optional
        image_dir (string): Image directory name, default value is 'img'

    Returns:
        string: Image folder directory or file path

    Example:
        >>> get_image_folder()
        'img'
        >>> get_image_folder(image_dir='image')
        'image'
        >>> get_image_folder(image_fn='chart.png')
        'img/chart.png'
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    return os.path.join(image_dir, image_fn) if image_fn else image_dir

def get_src_from_data(data=None, html=False):
    """Return Base64-encoded file to apply HTML <img> tag

    Reference:
        https://mindtrove.info/

    Args:
        data (string): image file path
    
    Returns:
        Base64-encoded file value

    Example:
        >>> get_src_from_data('image/chart.pnt')
    """
    if data:
        img_obj = Image(data=data)
        for bundle in img_obj._repr_mimebundle_():
            for mimetype, b64value in bundle.items():
                if mimetype.startswith('image/'):
                    encoded = f'data:{mimetype};base64,{b64value}'
                    if html:
                        return f'<img src="{encoded}">'
                    else:
                        return encoded
    else:
        if html:
            return "<span><i>Image Not Available</i></span>"
        else:
            return None

def get_variable_types(df):
    """Return variable type dictionary of data frame

    Example:
        >>> exp.get_variable_types(df_stock)
        {'Numeric': 6, 'Object': 0, 'Category': 0, 'Datetime': 1, 'Timedeltas': 0}
    """
    
    variable_types = {'Numeric': np.number,
                     'Object': 'object', 'Category': 'category',
                     'Datetime': 'datetime', 'Timedeltas': 'timedelta'}
    return {k: len(df.select_dtypes(include=v).columns) for k, v in variable_types.items()}

def get_freedman_bins(data, returnas="width", max_bins=50):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 

    http://www.jtrive.com/determining-histogram-bin-width-using-the-freedman-diaconis-rule.html

    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    Returns: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR = stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    N = data.size
    bw = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        if bw == 0:
            bw = 1
        result = int((datrng / bw) + 1)
        result = max_bins if result > max_bins else result
    return result
    
def get_variable_stat(df, col, save_hist=True, **kwargs):
    """
    Examples:
        >>> exp.get_variable_stat(df_stock, col='High', save_hist=True)
        {'variable_name': 'High',
        'dtype_str': 'float64',
        'size': 5031,
        'distinct': 4462,
        'distinct(%)': 88.69,
        ...
    """
    if df.empty:
        return None
    
    var_stat = dict()
    length = len(df)
    
    var_stat['variable_name'] = col
    var_stat['dtype_str'] = str(df[col].dtype)
    var_stat['size'] = length
    
    f2 = lambda x: float(int(x * 1000))/1000

    # quantile/descriptive statistics
    var_stat['distinct'] = len(df[col].unique())
    var_stat['distinct(%)'] = f2(var_stat['distinct'] / length * 100)
    var_stat['missing'] = df[col].isna().sum()
    var_stat['missing(%)'] = f2(var_stat['missing'] / length * 100)

    # datetime values
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        var_stat['minimum'] = df[col].min()
        var_stat['maximum'] = df[col].max()

    # numeric values
    if pd.api.types.is_numeric_dtype(df[col]):
        var_stat['minimum'] = df[col].min()
        var_stat['maximum'] = df[col].max()
        var_stat['special_char'] = sum(df[col] == '_')
        var_stat['zeros'] = sum(df[col] == 0)
        var_stat['zeros(%)'] = f2(var_stat['zeros'] / length * 100)
        var_stat['mean'] = df[col].mean()
        var_stat['median'] = df[col].median()
        var_stat['quantiles'] = dict(zip(['minimun', '5-th per.', 'Q1', 'median', 'Q3', '95-th per.', 'maximum'], list(df[col].quantile([0, .05, .25, .50, .75, .95, 1]).values)))
        var_stat['sum'] = df[col].sum()
        var_stat['sd'] = df[col].std()
        var_stat['skewness'] = df[col].skew()

    # common values
    value_counts = (pd.concat([
                        df[col].value_counts().rename('count'), 
                        df[col].value_counts(normalize=True).map(lambda x: float(int(x * 1000))/1000).rename('frequency (%)') * 100], axis=1)
                   .reset_index().rename(columns={'index':'value'}).reset_index(drop=True))
    
    var_stat['common_values'] = value_counts.head(6).values.tolist()
    var_stat['common_values'].insert(0, ['value', 'count', 'ratio(%)'])
    
    # extreme values
    if pd.api.types.is_numeric_dtype(df[col]):
        var_stat['min_extreme_values'] = list(value_counts.sort_values(by='value').head(6).values)
        var_stat['max_extreme_values'] = list(value_counts.sort_values(by='value').tail(6).values)
        var_stat['max_extreme_values'].insert(0, ['value', 'count', 'ratio(%)'])
        var_stat['min_extreme_values'].insert(0, ['value', 'count', 'ratio(%)'])

    image_dir = kwargs.get('image_dir', 'img')

    # histogram for numerical
    if save_hist and pd.api.types.is_numeric_dtype(df[col]):
        if df[col].dropna().empty:
            var_stat['image_fn'] = None
        else:
            freedman_bins = get_freedman_bins(data=df[col].dropna().values, returnas="bins")
            bins = kwargs.get('bins', freedman_bins)
            figsize = kwargs.get('figsize', (4,3))
            var_stat['image_fn'] = get_image_folder(f'hist_{col}_bins_{bins}_{figsize[0]}_{figsize[1]}.png')   
            ax = df[col].plot(kind='hist', bins=bins, figsize=figsize, edgecolor='black')
            ax.figure.savefig(var_stat['image_fn'])
            plt.close(ax.figure)
        
    # bar chart for object/categorical
    if save_hist and pd.api.types.is_object_dtype(df[col]):
        val_cnt = df[col].value_counts(ascending=False)
        len_values = len(val_cnt)
        show_bars_cnt = 5

        main_values = val_cnt
        if len_values > show_bars_cnt:
            main_values = val_cnt[:show_bars_cnt-1]
            other_value, other_len = val_cnt[show_bars_cnt-1:].sum(), len(val_cnt[show_bars_cnt-1:])
            main_values[f'Others ({other_len})'] = other_value
        
        # bar chart from main_values
        figsize = kwargs.get('figsize', (7,3))
        var_stat['image_fn'] = get_image_folder(f'hbar_{col}_{figsize[0]}_{figsize[1]}.png') 
        ax = main_values[::-1].plot.barh(figsize=figsize)
        ax.figure.savefig(var_stat['image_fn'])
        plt.close(ax.figure)
        
        # pie chart from val_cnt
        pie_figsize = (3,3)
        var_stat['image_fn2'] = get_image_folder(f'pie_{col}_{figsize[0]}_{figsize[1]}.png') 
        ax_pie = val_cnt.plot.pie(figsize=pie_figsize)
        ax_pie.figure.savefig(var_stat['image_fn2'])
        plt.close(ax_pie.figure)

    # bar chart for datetime
    if save_hist and pd.api.types.is_datetime64_any_dtype(df[col]):
        if df[col].dropna().empty:
            var_stat['image_fn'] = None
        else:
            freedman_bins = get_freedman_bins(data=df[col].dropna().values, returnas="bins")
            bins = kwargs.get('bins', freedman_bins)
            figsize = kwargs.get('figsize', (8, 3))
            var_stat['image_fn'] = get_image_folder(f'hist_{col}_bins_{bins}_{figsize[0]}_{figsize[1]}.png')   
            ax = df[col].astype(np.int64).plot(kind='hist', bins=bins, figsize=figsize, edgecolor='black')

            labels = ax.get_xticks().tolist()
            ax.set_xticks(labels)
            ax.set_xticklabels(pd.to_datetime(labels).strftime('%Y-%m-%d'))

            ax.figure.savefig(var_stat['image_fn'])
            plt.close(ax.figure)

    return var_stat    
    
def plot_interface(func):
    def wrapper(*args, **kwargs):
        bypass = kwargs.get('bypass', False)
        return_ax = kwargs.get('return_ax', False)
        save_only = kwargs.get('save_only', False)
        image_fn = kwargs.get('image_fn', get_image_folder(f'{func.__name__}_{args[1]}_{str(uuid.uuid4())}.png'))
        figsize = kwargs.get('figsize', (7,5))
        
        if func.__name__ not in ['plot_cat2cat', 'plot_num2hist']:
            plt.figure(figsize=figsize)

        ax, df = func(*args, **kwargs)
                
        if return_ax:
            return ax

        if save_only:
            if hasattr(ax, "__len__"):
                plt.savefig(image_fn)
                for ax_row in ax:
                    for ax_item in ax_row:
                        plt.close(ax_item.figure)
            else:
                ax.figure.savefig(image_fn)
                plt.close(ax.figure)
            return image_fn

        if bypass:
            return df
        
    return wrapper

@plot_interface
def plot_num2num(df, num_col, target_col, hue=None, style=None, **kwargs):
    '''
    Example:
        >>> plot_num2num(df_dia_train, 'carat', 'price', hue='color', style='cut', save_only=False, figsize=(12,8))
        >>> plot_num2num(df_dia_train, 'carat', 'price', hue='color', style='cut', bypass=True, figsize=(12,8))
    '''
    ax = sns.scatterplot(data=df, x=num_col, y=target_col, hue=hue, style=style)
    return ax, df

@plot_interface
def plot_cat2num(df, cat_col, target_col, hue=None, **kwargs):
    '''
    Example:
        >>> plot_cat2num(df_dia_train, 'color', 'price', save_only=True)
        >>> plot_cat2num(df_dia_train, 'color', 'price', hue='cut', figsize=(15, 8))
    '''
    ax = sns.boxplot(data=df, x=cat_col, y=target_col, hue=hue)
    return ax, df

@plot_interface
def plot_num2hist(df, num_col, cat_cols=None, **kwargs):
    '''
    Example:
        >>> plot_num2hist(df_titanic, ['Survived', 'Pclass'], 'Age')
    '''
    figsize = kwargs.get('figsize', (6,4))
    width, height = figsize[0], figsize[1]
    aspect = width / height
    
    freedman_bins = get_freedman_bins(data=df[num_col].dropna().values, returnas="bins")
    bins = kwargs.get('bins', freedman_bins)

    if cat_cols == None:
        ax = df[num_col].plot(kind='hist', alpha=.5, bins=bins, figsize=figsize)
    else:
        row = None
        col = cat_cols[0]

        if len(cat_cols) == 2:
            row = cat_cols[1]

        g = sns.FacetGrid(df, col=col, row=row, height=height, aspect=aspect)
        g.map(plt.hist, num_col, alpha=.5, bins=bins)
        g.add_legend();
        ax = g.axes if col else g.ax
            
    return ax, df

@plot_interface
def plot_cat2pie(df, cat_cols, **kwargs):
    '''
    Example:
        >>> plot_cat2pie(df_titanic, 'Pclass', figsize=(3,3))
    '''
    figsize = kwargs.get('figsize', (6,4))
    ax = df[cat_cols].value_counts().plot.pie(figsize=figsize)
    return ax, df

@plot_interface
def plot_cat2cat(df, cat_col, target_col, col=None, col_wrap=None, normalized=False, **kwargs):
    '''
    Example:
        >>> plot_cat2cat(df_titanic_train, 'Pclass', 'Survived', col='Sex')
        >>> plot_cat2cat(df_titanic_train, 'Pclass', 'Survived')
        >>> plot_cat2cat(df_titanic_train, 'Pclass', 'Survived', normalized=True)
        >>> plot_cat2cat(df_titanic_train, 'Pclass', 'Survived', normalized=False, save_only=False)
    '''
    figsize = kwargs.get('figsize', (6,4))
    width, height = figsize[0], figsize[1]
    aspect = width / height
    
    if normalized:
        ax = (df
         .pipe(lambda df: pd.crosstab(df[cat_col], df[target_col]))
         .pipe(lambda df: df.div(df.sum(axis=1), axis=0))
         .plot(kind='bar', stacked=True, figsize=figsize)
        )   
    else:
        g = sns.catplot(data=df, x=cat_col, kind='count', hue=target_col, col=col, col_wrap=col_wrap, height=height, aspect=aspect)
        ax = g.axes if col else g.ax
        
    return ax, df

@plot_interface
def plot_num2cat(df, cat_col, target_col, multiple='layer', **kwargs):
    '''
    Example:
        >>> plot_n2c(df_titanic, 'Age', 'Survived', multiple='fill')
        >>> plot_n2c(df_titanic, 'Age', 'Survived', multiple='layer')
    '''
    ax = sns.kdeplot(data=df, x=cat_col, hue=target_col, multiple=multiple)
    return ax, df

def plot_cats2binary(df, cat_cols, target_col, hue=None, **kwargs):
    '''
    Example:
        1) category nums == 3
        >>> plot_cats2binary(df_titanic, ['Pclass', 'Sex', 'Embarked'], 'Survived')
        2) category nums == 2
        >>> plot_cats2binary(df_titanic, ['Pclass', 'Sex'], 'Survived')
        3) category nums == 1
        >>> plot_cats2binary(df_titanic.assign(relatives = lambda df: df.Parch + df.SibSp), ['relatives'], 'Survived')
    '''
    assert(len(df[target_col].unique()) == 2)
    bypass = kwargs.get('bypass', False)
    figsize = kwargs.get('figsize', (6,4))
    width, height = figsize[0], figsize[1]
    aspect = width / height

    # len(cat_cols) == 3
    if len(cat_cols) == 3:
        FacetGrid = sns.FacetGrid(df, col=cat_cols[2], height=height, aspect=aspect)
        FacetGrid.map(sns.pointplot, cat_cols[0], target_col, cat_cols[1], palette=None, order=None, hue_order=None )
        FacetGrid.add_legend()

    # len(cat_cols) == 2
    if len(cat_cols) == 2:
        sns.pointplot(data=df, x=cat_cols[0], y=target_col, hue=cat_cols[1], height=height, aspect=aspect)

    # len(cat_cols) == 1
    if len(cat_cols) == 1:
        sns.catplot(data=df, x=cat_cols[0], y=target_col, kind="point", hue=hue, height=height, aspect=aspect)
        
    plt.show()
    
    if bypass:
        return df
    
def render_cat2cat(df, cat_col, target_col):
    '''
    Example:
        >>> render_cat2cat(df_titanic_train, 'Pclass', 'Survived')
    '''
    cat2cat_counts = plot_cat2cat(df, cat_col, target_col, figsize=(5,3), save_only=True)
    cat2cat_normalized = plot_cat2cat(df, cat_col, target_col, normalized=True, figsize=(5,3), save_only=True)
    header_style = 'style="font-weight:bold;text-align:center;font-size:1.3em"'
    col_header_style = 'style="font-weight:bold;text-align:center;font-size:1.2em;background-color:white"'

    html = f'''
        <table>
            <tr><td colspan=2 {header_style}>"{target_col}" by "{cat_col}"</td></tr>
            <tr><td {col_header_style}>Class Counts</td><td {col_header_style}>Normalized Counts</td></tr>
            <tr>
                <td><img src="{get_src_from_data(cat2cat_counts)}"></td>
                <td><img src="{get_src_from_data(cat2cat_normalized)}"></td>
            </tr>
        </table>
    '''
    display(HTML(html))

def render_num2cat(df, num_col, target_col):
    '''
    Example:
        >>> render_num2cat(df_titanic_train, 'Age', 'Survived')
    '''
    layer_image = plot_num2cat(df, num_col, target_col, multiple='layer', figsize=(5,3), save_only=True)
    fill_image = plot_num2cat(df, num_col, target_col, multiple='fill', figsize=(5,3), save_only=True)
    # unique 숫자에 따른 변수의 분포에 대한 이미지 생성해서 반영

    header_style = 'style="font-weight:bold;text-align:center;font-size:1.3em"'
    col_header_style = 'style="font-weight:bold;text-align:center;font-size:1.2em;background-color:white"'

    html = f'''
        <table>
            <tr><td colspan=2 {header_style}>"{target_col}" by "{num_col}"</td></tr>
            <tr><td {col_header_style}>Class Distribution</td><td {col_header_style}>Normalized Distribution</td></tr>
            <tr>
                <td><img src="{get_src_from_data(layer_image)}"></td>
                <td><img src="{get_src_from_data(fill_image)}"></td>
                <!-- 이미지 생성한 것 반영 -->
            </tr>
        </table>
    '''
    display(HTML(html))
    
def render_num2num(df, num_col, target_col, **kwargs):
    '''
    Example:
        >>> render_num2num(df_stock, num_col='Open', target_col='Close')
    '''
    hue = kwargs.get('hue', None)
    style = kwargs.get('style', None)

    image_default = plot_num2num(df, num_col, target_col, figsize=(5,4), save_only=True)
    image_category = plot_num2num(df, num_col, target_col, hue=hue, style=style, figsize=(5,4), save_only=True)
    image_hist = plot_num2hist(df, num_col, figsize=(5,4), save_only=True)

    header_style = 'style="font-weight:bold;text-align:center;font-size:1.3em"'
    col_header_style = 'style="font-weight:bold;text-align:center;font-size:1.2em;background-color:white"'
    
    html = f'''
        <table>
            <tr><td {header_style} colspan=3>"{target_col}" Distribution by "{num_col}" </td></tr>
            <tr>
                <td><img src="{get_src_from_data(image_default)}"></td>
                <td><img src="{get_src_from_data(image_category)}"></td>
                <td><img src="{get_src_from_data(image_hist)}"></td>
            </tr>
        </table>
    '''
    display(HTML(html))
    
def render_cat2num(df, cat_col, target_col, **kwargs):
    '''
    Example:
        >>> render_num2num(df_stock, num_col='Weekday', target_col='Close')
    '''
    image_box = plot_cat2num(df, cat_col, target_col, figsize=(7,4), save_only=True)
    image_pie = plot_cat2pie(df, cat_col, figsize=(7,4), save_only=True)

    header_style = 'style="font-weight:bold;text-align:center;font-size:1.3em"'
    col_header_style = 'style="font-weight:bold;text-align:center;font-size:1.2em;background-color:white"'
    
    html = f'''
        <table>
            <tr><td {header_style} colspan=2>"{target_col}" Distribution by "{cat_col}" </td></tr>
            <tr>
                <td><img src="{get_src_from_data(image_box)}"></td>
                <td><img src="{get_src_from_data(image_pie)}"></td>
            </tr>
        </table>
    '''
    display(HTML(html))

def render_cols_table(*args, title=None, render=True):
    """1개의 행에 여러개의 열 정보를 구성할 때 사용하는 html 렌더링 함수"""
    table = '<table style="background-color:white;"><tr><td style="vertical-align:top;text-align:left">{}</td></tr></table>'.format('</td><td style="vertical-align:top;text-align:left">'.join(str(cell).replace("['", "").replace("']", "").replace("', '", "") for cell in args))

    if title and render:
        display(Markdown(title))
        
    if render:
        display(HTML(table))
        
    if render:
        return None
    else:
        title = markdown2.markdown(title) if title else ''
        return (title + table).replace('\n', '')        

def render_dataset_info(df, df2=None, set_name=None, df_name=None, df2_name=None, **kwargs):
    '''
    Example:
        >>> render_dataset_info(dfs=[df_titanic_train], df_names=['Train'], set_name='Titanic')
        >>> render_dataset_info(dfs=[df_titanic_train, df_titanic_test], df_names=['Train', 'Test'], set_name='Titanic')
    '''

    title = kwargs.get('title', f"## {set_name} Datasets")

    if df2 is not None:
        render_cols_table(
            render_dict2table(get_dataset_stat(df), title=f'{df_name} statistics', render=False), 
            render_dict2table(get_variable_types(df), title=f'variable types', render=False),
            render_dict2table(get_dataset_stat(df2), title=f'{df2_name} statistics', render=False),
            render_dict2table(get_variable_types(df2), title=f'variable types', render=False), 
            render=True, title=title)
    else:
        render_cols_table(
            render_dict2table(get_dataset_stat(df), title=f'{df_name} statistics', render=False), 
            render_dict2table(get_variable_types(df), title=f'variable types', render=False),
            render=True, title=title)

def render_variables_info(df, set_name=None, **kwargs):
    '''
    Example:
        >>> render_variable_info(df_titanic, set_name='Titanic')
    '''
    title = kwargs.get('title', f"## {set_name} Variables")
    display(Markdown(title))
    for idx, col in enumerate(df.columns.values):
        display(Markdown(f'### {idx+1}. "{col}"'))
        render_variable(df, col)
        display(Markdown(f'----------'))

def render_target_by_feature_clf(df, target, set_name=None, **kwargs):
    '''
    Example:
        >>> render_target_by_feature_clf(df_titanic, target='Survived', set_name='Titanic', title='Titanic Variable Analysis')
    '''
    title = kwargs.get('title', f"## {set_name} Target Distributiones")
    display(Markdown(title))

    for col in df.drop(columns=[target]).dtypes.sort_values().index:
        if pd.api.types.is_numeric_dtype(df[col]):
            render_num2cat(df, col, target)

    for col in df.drop(columns=[target]).dtypes.sort_values().index:
        if pd.api.types.is_object_dtype(df[col]):
            render_cat2cat(df, col, target)
            
def render_target_by_feature_reg(df, target, set_name=None, **kwargs):
    '''
    Example:
        >>> render_target_by_feature_reg(df_stock, target='Close', set_name='Stock', title='Stock Variable Analysis')
    '''
    title = kwargs.get('title', f"## {set_name} Target Distributiones")
    display(Markdown(title))

    for col in df.drop(columns=[target]).dtypes.sort_values().index:
        if pd.api.types.is_numeric_dtype(df[col]):
            n2n_option = kwargs.get('n2n_option', {'hue': None, 'style': None})
            render_num2num(df, col, target, hue=n2n_option['hue'], style=n2n_option['style'], figsize=(6,4))

    for col in df.drop(columns=[target]).dtypes.sort_values().index:
        if pd.api.types.is_object_dtype(df[col]):
            render_cat2num(df, col, target)

def render_dict2table(dict_data, header='col', floatfmt=None, title=None, render=None, selected_keys=None):
    """
    Example:
        >>> data_stat = exp.dataset_stat(df_stock)
        >>> HTML(exp.render_dict2table(data_stat, floatfmt=[None, None, None, 2, None, 2], title='Data Statistics'))
        Data Statistics
        Number of variables	7
        Number of observations	5031
        Missing cells	0
        Missing cells(%)	0.00
        Duplicated rows	0
        Duplicated rows(%)	0.00
    """
    if selected_keys:
        dict_data = {k: v for k, v in dict_data.items() if k in selected_keys}

    return render_list2table([[k, v] for k, v in dict_data.items()], header=header, floatfmt=floatfmt, title=title, render=render)

def render_list2table(list_data, header=None, floatfmt=None, title=None, render=None):
    """리스트 자료형에서 html 태그를 생성. 컬럼해더 및 소수점 지정 가능"""

    html = '<table style="background-color:white">'
    if title is not None:
        colspan = len(list_data[0])
        header_style = 'style="font-weight:bold;text-align:center;font-size:1.2em"'
        html += f'<tr><td colspan={colspan} {header_style}>{title}</td></tr>'

    for rid, row in enumerate(list_data):
        html += '<tr>'
        for cid, cell in enumerate(row):
            td_style = ''
            if (header == 'col' and cid == 0) or (header == 'row' and rid == 0):
                td_style = 'style="font-weight:bold"'

            value = cell

            # 시간 날 때 리팩토링 할 함수
            if header == 'row' and floatfmt and type(cell) != str:
                if floatfmt[cid] == 0:
                    value = f'{cell:.0f}'
                if floatfmt[cid] == 2:
                    value = f'{cell:.2f}'
                if floatfmt[cid] == 4:
                    value = f'{cell:.4f}'

            if header == 'col' and floatfmt and type(cell) != str:
                if floatfmt[rid] == 0:
                    value = f'{cell:.0f}'
                if floatfmt[rid] == 2:
                    value = f'{cell:.2f}'
                if floatfmt[rid] == 4:
                    value = f'{cell:.4f}'
                    
            html += f'<td {td_style}>{value}</td>'
        html += '</tr>'
    html += '</table>'

    if render:
        display(HTML(html))
    else:
        return html            

def render_variable(df, col, title=False):
    '''
    Example:
        >>> render_variable(df_titanic, 'Pclass')
    '''
    content = get_variable_stat(df, col)
    
    html = '<i>empty</i>'
    header_style = 'style="font-weight:bold;text-align:center;font-size:1.2em"'

    # for numerical
    if pd.api.types.is_numeric_dtype(df[col]):
        col1 = render_dict2table(content, render=False, selected_keys=['size', 'distinct', 'distinct(%)', 'missing', 'missing(%)', 'minimum', 'maximum'], header='col', floatfmt=[0, 0, 2, 0, 2, None, None])
        col2 = render_dict2table(content, render=False, selected_keys=['mean', 'median', 'sum', 'sd', 'skewness', 'zeros', 'zeros(%)'], header='col', floatfmt=[0, 2, 2, 2, 2, 2, 2])
        col3 = render_dict2table(content['quantiles'], header='col', render=False, floatfmt=[None, 2, 2, None, 2, 2, None])
        image_tag = get_src_from_data(content['image_fn'], html=True)
        html = f'''
        <table style="background-color:white">
            <tr>
                <td colspan=2 {header_style}>| Descriptive Stat. |</td>
                <td {header_style}>| Quantile Stat. |</td>

                <td {header_style}>| Common Values |</td>
                <td {header_style}>| Min Values |</td>
                <td {header_style}>| Max Values |</td>
                <td {header_style}>| Histogram |</td>
            </tr>
            <tr>
                <td>{col1}</td>
                <td>{col2}</td>
                <td>{col3}</td>
                <td>{render_list2table(content['common_values'], header='row', floatfmt=[None, 0, 2])}</td>
                <td>{render_list2table(content['min_extreme_values'], header='row', floatfmt=[None, 0, 2])}</td>
                <td>{render_list2table(content['max_extreme_values'], header='row', floatfmt=[None, 0, 2])}</td>
                <td>{image_tag}</td>
            </tr>
        </table>
        '''.replace('\n', '')
    
    # for object/categorical
    if pd.api.types.is_object_dtype(df[col]):
        col1 = render_dict2table(content, render=False, selected_keys=['size', 'distinct', 'distinct(%)', 'missing', 'missing(%)'], header='col')
        image_tag = get_src_from_data(content['image_fn'], html=True)
        image_tag2 = get_src_from_data(content['image_fn2'], html=True)

        html = f'''
        <table>
            <tr>
                <td {header_style}>| Descriptive Statistics |</td>
                <td {header_style}>| Common Values |</td>
                <td {header_style}>| Bar Chart |</td>
                <td {header_style}>| Pie Chart |</td>
            </tr>
            <tr>
                <td>{col1}</td>
                <td>{render_list2table(content['common_values'], header='row', floatfmt=[None, 0, 2])}</td>
                <td>{image_tag}</td>
                <td>{image_tag2}</td>
            </tr>
        </table>
        '''.replace('\n', '')

    # for datetime/timedelta
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        col1 = render_dict2table(content, render=False, selected_keys=['size', 'distinct', 'distinct(%)', 'missing', 'missing(%)', 'minimum', 'maximum'], header='col')
        image_tag = get_src_from_data(content['image_fn'], html=True)

        html = f'''
        <table>
            <tr>
                <td {header_style}>| Descriptive Statistics |</td>
                <td {header_style}>| Common Values |</td>
                <td {header_style}>| Pie Chart |</td>
            </tr>
            <tr>
                <td>{col1}</td>
                <td>{render_list2table(content['common_values'], header='row', floatfmt=[None, 0, 2])}</td>
                <td>{image_tag}</td>
            </tr>
        </table>
        '''.replace('\n', '')
    
    if title:
        html = f'<h2>{title}</h2>' + html
        
    # html
    display(HTML(html))    
    
def report_dataset(df, df_name, set_name, target_col, model_type='clf'):
    '''
    Example:
        >>> report_dataset(df_titanic, df_name='Titanic', set_name='Train', target_col='Survived', model_type='clf')
    '''
    # dataset info
    render_dataset_info(df, df_name=df_name, set_name=set_name, title=f'## 1. "{set_name}" Dataset Information')

    # variable info
    render_variables_info(df, set_name=set_name, title=f'## 2. "{set_name}" Variables')

    # target distribution by feature
    title_feature_dist = f'## 3. "{set_name}" Target({target_col}) Distribution'
    if model_type == 'reg':
        render_target_by_feature_reg(df, target_col, set_name=set_name, title=title_feature_dist)
    elif model_type == 'clf':
        render_target_by_feature_clf(df, target_col, set_name=set_name, title=title_feature_dist)