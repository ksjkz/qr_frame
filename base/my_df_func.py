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
  df=load_df(df,groupby_column=ticker_column,sort_column=time_column)
  df[return_column]=df[close_column].pct_change(-1)
  indices = df.groupby(ticker_column).apply(lambda x: x.tail(1).index).explode().values
  df.loc[indices,return_column]=None
  return df



def linear_reg(df, f: str, r:str,is_print=False,is_plot=False):
    # 创建线性回归模型
    model = LinearRegression()
    df_clean = df[[f, r]].dropna()
    # 重新定义 X 和 y，确保没有 NaN 值，并且长度一致
    X_clean = df_clean[[f]]  # 二维数据
    y_clean = df_clean[r]    # 一维目标数据
    lenth = X_clean.shape[0]
    # 拟合线性回归模型
    model.fit(X_clean, y_clean)
    # 使用模型预测结果
    y_pred = model.predict( X_clean)

    # 计算 R²
    r2 = r2_score(y_clean, y_pred)
    slope = model.coef_[0]  # 斜率
    intercept = model.intercept_  # 截距
    if is_print:
        print(f"数据点对数：{lenth}")
        print(f"R²: {r2}")
        print(f"斜率 (Slope): {slope}")
        print(f"截距 (Intercept): {intercept}")

    if is_plot:
       

                 plt.figure(figsize=(8, 6))
                 sns.lineplot(x=X_clean.iloc[:, 0], y=y_pred, color='blue', label='fitting line (yhat)')
                 sns.scatterplot(x=X_clean.iloc[:, 0], y=y_clean, color='red', label='original data (y)')
                 plt.legend()
                 plt.title('linear regression and law data point')
                 plt.xlabel('X')
                 plt.ylabel('y')
                 buf = BytesIO()
                 plt.savefig(buf, format='png')
                 buf.seek(0)  # 将指针移到文件的开头
                 plt.close()
                 display(Image(data=buf.getvalue()))
                 buf.close()
       
    return r2,lenth,slope,intercept
   

