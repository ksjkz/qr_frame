from base import*
from sklearn.preprocessing import StandardScaler
import pandas as pd
from base.formulate_coding_AST import load_df


def group_std(df:pd.DataFrame,columns_list:list,by:str='t_date',):
   '''
   对columns_list列进行group后再标准化
   输入 df
   columns_list:[str] str为列名
   by: 分组列名
   '''
   for column in columns_list:
     df[column] = df.groupby(by)[column].transform(lambda x: (x - x.mean()) / x.std())
   return df

def set_return(df:pd.DataFrame,time_column:str='t_date',ticker_column:str='ticker',close_column:str='close',return_column:str='one_day_return'):
  '''
  为df添加一列return
  因为要通过load_df,所以输入无需排序
  '''
  df=load_df(df,groupby_column=ticker_column,sort_column=time_column)
  df[return_column]=df[close_column].pct_change(-1)
  indices = df.groupby(ticker_column).apply(lambda x: x.tail(1).index).explode().values
  df.loc[indices,return_column]=None
  return df




