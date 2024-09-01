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

def group_std(df:pd.DataFrame,columns_list:list,by:str='T_DATE',):
   '''
   对columns_list列进行group后再标准化
   '''
   def standardize_group(group):
    scaler = StandardScaler()
    group[columns_list] = scaler.fit_transform(group[columns_list])
    return group
   d3 = df.groupby(by).apply(standardize_group)
   return d3