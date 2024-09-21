from base import*
from sklearn.preprocessing import StandardScaler
import pandas as pd
from base.formulate_coding_AST import load_df
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from IPython.display import display, Image
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
  df1=load_df(df,groupby_column=ticker_column,sort_column=time_column)
  df1['next_day_close']=df1[close_column].shift(-1)
  df1[return_column]=(df1['next_day_close']-df1[close_column])/df1[close_column]
  df1.drop(columns=['next_day_close'],inplace=True)
  indices = df1.groupby(ticker_column).apply(lambda x: x.tail(1).index).explode().values
  df1.loc[indices,return_column]=None
  return df1



def linear_reg(df, f: str, r:str,is_print=False,):
    '''
    线性回归
    df: pd.DataFrame
    f: str  用于回归的列1
    r: str   用于回归的列2
    is_print: bool

    返回值:
    r2: float
    lenth: int 回归数据的长度
    slope: float
    intercept: float
    '''
    model = LinearRegression()
    df_clean = df[[f, r]].dropna()# 重新定义 X 和 y，确保没有 NaN 值，并且长度一致
    X_clean = df_clean[[f]]  # 二维数据
    y_clean = df_clean[r]    # 一维目标数据
    lenth = X_clean.shape[0]
    model.fit(X_clean, y_clean)
    y_pred = model.predict( X_clean)
    r2 = r2_score(y_clean, y_pred)
    slope = model.coef_[0]  # 斜率
    intercept = model.intercept_  # 截距
    if is_print:
        print(f"数据点对数：{lenth}")
        print(f"R²: {r2}")
        print(f"斜率 (Slope): {slope}")
        print(f"截距 (Intercept): {intercept}")
       
    return r2,lenth,slope,intercept
# 定义一个函数来去除极值
def remove_df_outliers(df:pd.DataFrame,col:str,by:str='t_date',):
    '''
    对于单列，groupby后去除极值
    如果不用groupby，请将by设置为''
    返回去除极值后的df(不会对原来df进行修改)
    '''
    def remove_outliers(group):
         Q1 = group[col].quantile(0.25)  # 第1四分位数
         Q3 = group[col].quantile(0.75)  # 第3四分位数
         IQR = Q3 - Q1  # 四分位距
         lower_bound = Q1 - 1.5 * IQR  # 下界
         upper_bound = Q3 + 1.5 * IQR  # 上界
         return group[(group[col] >= lower_bound) & (group[col] <= upper_bound)]
    if by == '':
        return df.apply(remove_outliers).reset_index(drop=True)
    return df.groupby(by).apply(remove_outliers).reset_index(drop=True)
       



   

