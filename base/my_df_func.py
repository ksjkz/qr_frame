from typing import List, Dict
import warnings
import pandas as pd
import numpy as np
from loguru import logger
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import datetime
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts import options as opts
from scipy.stats import rankdata
from wcorr import WeightedCorr
from statsmodels.tsa.stattools import acf


def group_std(df:pd.DataFrame,columns_list:list,by:str='t_date',):
   '''
   对columns_list列进行group后再标准化
   输入 df
   columns_list:[str] str为列名
   by: 分组列名
   '''
   def standardize_group(group):
    scaler = StandardScaler()
    group[columns_list] = scaler.fit_transform(group[columns_list])
    return group
   d3 = df.groupby(by).apply(standardize_group)
   return d3

from base.formulate_filter_AST import load_df

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




