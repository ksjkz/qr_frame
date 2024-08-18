import pandas as pd
from tqdm import tqdm
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
from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts import options as opts
from scipy.stats import rankdata
from wcorr import WeightedCorr
from statsmodels.tsa.stattools import acf


params = {
    't_date': 'T_DATE',
    'ticker': 'TICKER',
    '_return': 'ONE_DAY_RETURN',
    'weight': 'MCAP_IN_CIRCULATION',
    '_return_prev':'ONE_DAY_RETURN_PREV',
    '_return_prev1':'D_CLOSE_CHANGE',
    '_weight':'LOG_MCAP'
}
exclude_columns=[ 'ADJ_D_CLOSE','ADJ_D_OPEN','ADJ_D_HIGH','ADJ_D_LOW','YUAN_VOLUME','MCAP_IN_CIRCULATION','CLASS_NAME','D_OPEN_CHANGE', 'D_HIGH_CHANGE', 'D_LOW_CHANGE', 'D_CLOSE_CHANGE','D_VOLUME_CHANGE', 'D_PB_CHANGE', 'LOG_MCAP',*(params.values())]


class Eval:
    
    '''
    类名：Eval
------------------输入参数
                        ---df 包含所有信息的完整的df(注意初始化的时候会自动将df进行dropna)
                        ---factor_name 因子名称 为df的某一列的列名
                        ---outlier_removal_opt 是否去极值 一般为True
                        ---std_opt 是否标准化 一般为TRue

--------------------方法:
                        --- cul() 
                                  计算因子的各种评价指标,并将其变成类的属性方便调用
                          quota_dict:包含日期 iC,rankic,和icir的df
                          x:日期组成的list
                          IC_list
                          rankIc_list
                          P_value_list
                          IC_mean
                          rankIc_mean
                          P_value_mean
                          ICIR
                          autocorr:从滞后1到10





'''
    def __init__(self,df:pd.DataFrame,factor_name:str='a',outlier_removal_opt=True,std_opt=True,selct_ticker_opt=True):
        self.df=df.dropna()

        if factor_name =='a': #判断有没有提供因子列的列名,如果没有,则自动选择因子列
            remaining_columns = [col for col in self.df.columns if col not in exclude_columns]
            self.factor_name=  remaining_columns[0]
        else:   
             self.factor_name=factor_name

        if outlier_removal_opt:#去极值 四分位法
             Q1 = self.df[self.factor_name].quantile(0.25)
             Q3 = self.df[self.factor_name].quantile(0.75)
             IQR = Q3 - Q1
             lower_bound = Q1 - 1.5 * IQR
             upper_bound = Q3 + 1.5 * IQR
             self.df =self.df[(self.df[self.factor_name] >= lower_bound) & (self.df[self.factor_name] <= upper_bound)]

        if std_opt:#标准化
            for name,group in self.df.groupby(params['t_date']):
                a=group[self.factor_name].mean()
                b=group[self.factor_name].std()
                a1=group[params['_return']].mean()
                b1=group[params['_return']].std()
                self.df.loc[group.index, self.factor_name] =group[self.factor_name].apply(lambda x: (x - a) / b)
                self.df.loc[group.index, params['_return']]=group[params['_return']].apply(lambda x: (x - a1) / b1)
        
        if selct_ticker_opt:#选择股票池  市值小于30亿,成交额小于 1kw,drop,涨跌停,drop
            if params['_return_prev'] in self.df.columns:
                  self.df=self.df[(self.df[params['weight']]<3000000000)&(self.df['YUAN_VOLUME']>1000000)&(self.df[params['_return_prev']]<0.099)&(self.df[params['_return_prev']]>-0.099)]
            else:
                  self.df=self.df[(self.df[params['_weight']]<21.82)&(self.df[params['_return_prev1']]<0.099)&(self.df[params['_return_prev1']]>-0.099)]


        self.flag=False#用于提示画图前有没有经过计算

        if params['weight'] in self.df.columns:
            self.df['CAP_WEIGHT']=np.log(self.df[params['weight']])#取对数权重
        else:
            self.df['CAP_WEIGHT']=self.df[params['_weight']]

        

    def cul(self):
        grouped_df=self.df.groupby(params['t_date'])
        aa={}
        for name,group in tqdm(grouped_df,leave=True,desc='正在回测'):
          
             WC = WeightedCorr(xyw=group[[self.factor_name,params['_return'],'CAP_WEIGHT']])
             ic=WC(method="pearson")
             rho=WC( method="spearman")
            
             p_value=0   #这里P值没有计算,因为计算量太大
             
             aa[name]=[ic,rho,p_value]
      
        x=list(aa.keys())
        y1=[value[0] for value in aa.values()][:-1]
        y2=[value[1] for value in aa.values()][:-1]
        y3=[value[2] for value in aa.values()][:-1]
        try:
               autocorr = acf(y1, nlags=len(y1)-1)[1:10]  #这个会返回从滞后0到滞后n-1的ACF值，所以取第二个 返回值为ndarray
               setattr(self, 'autocorr', list(autocorr))
        except:
                autocorr=0
                setattr(self, 'autocorr', autocorr)
        setattr(self, 'quota_dict', aa)
        setattr(self, 'x', x)
        setattr(self, 'IC_list', y1)
        setattr(self, 'rankIc_list', y2)
        setattr(self, 'P_value_list', y3)
        setattr(self, 'IC_mean', np.mean(y1))
        setattr(self, 'rankIc_mean', np.mean(y2))
        setattr(self, 'P_value_mean', np.mean(y3))
        ic_array = np.array(y2)
        ICIR=np.mean(ic_array)/np.std(ic_array)*np.sqrt(len(ic_array)-1)
        setattr(self, 'ICIR',ICIR )
        self.flag=True
        print('-------------------------因子回测完成-----------------------\n')

    

    def plot(self):
        if self.flag==False:
            self.cul()
        _line = (
                  Line()
                 .add_xaxis(self.x)
                 .add_yaxis("IC",self.IC_list,label_opts=opts.LabelOpts(is_show=False))
                 .add_yaxis("RankIc", self.rankIc_list,label_opts=opts.LabelOpts(is_show=False))
                 .add_yaxis("P_value", self.P_value_list,label_opts=opts.LabelOpts(is_show=False))
                  #.add_yaxis("值2", y2)
                 .set_global_opts(
                 title_opts=opts.TitleOpts(title=self.factor_name),
                 xaxis_opts=opts.AxisOpts(name="分类",axislabel_opts=opts.LabelOpts(rotate=45)),
                 yaxis_opts=opts.AxisOpts(name="数值",),
                 legend_opts=opts.LegendOpts(pos_top="top")
               ))
        _line.render(self.factor_name+'.html')

def weighted_correlation(x, y, weights):
    """
    计算两组数据之间的加权相关系数。

    参数:
    x (array-like): 第一个数据数组。
    y (array-like): 第二个数据数组。
    weights (array-like): 每个数据点的权重。

    返回:
    float: 加权相关系数。
    """
    # 归一化权重
    normalized_weights = weights / np.sum(weights)
    
    def weighted_mean(values, weights):
        return np.sum(values * weights) / np.sum(weights)

    def weighted_covariance(x, y, weights):
        mean_x = weighted_mean(x, weights)
        mean_y = weighted_mean(y, weights)
        return np.sum(weights * (x - mean_x) * (y - mean_y))

    def weighted_variance(values, weights):
        mean = weighted_mean(values, weights)
        return np.sum(weights * (values - mean)**2)

    cov_xy = weighted_covariance(x, y, normalized_weights)
    var_x = weighted_variance(x, normalized_weights)
    var_y = weighted_variance(y, normalized_weights)

    return cov_xy / np.sqrt(var_x * var_y)

def weighted_rank_correlation(x, y, weights, num_permutations=100):
    """
    计算两组数据之间的加权秩相关系数及其 p 值。

    参数:
    x (array-like): 第一个数据数组。
    y (array-like): 第二个数据数组。
    weights (array-like): 每个数据点的权重。
    num_permutations (int): 用于计算 p 值的排列次数，默认为 100。

    返回:
    tuple: 加权秩相关系数和 p 值。
    """
    def weighted_mean(values, weights):
        return np.sum(values * weights) / np.sum(weights)

    def weighted_covariance(x, y, weights):
        mean_x = weighted_mean(x, weights)
        mean_y = weighted_mean(y, weights)
        return np.sum(weights * (x - mean_x) * (y - mean_y)) / np.sum(weights)

    def weighted_variance(values, weights):
        mean = weighted_mean(values, weights)
        return np.sum(weights * (values - mean)**2) / np.sum(weights)

    # 将数据转换为秩
    rank_x = rankdata(x)
    rank_y = rankdata(y)

    # 归一化权重
    normalized_weights = weights / np.sum(weights)

    # 计算加权协方差
    cov_rank_xy = weighted_covariance(rank_x, rank_y, normalized_weights)

    # 计算加权方差
    var_rank_x = weighted_variance(rank_x, normalized_weights)
    var_rank_y = weighted_variance(rank_y, normalized_weights)

    # 计算加权秩相关系数
    weighted_rank_corr = cov_rank_xy / np.sqrt(var_rank_x * var_rank_y)

    # 计算 p 值
    permuted_corrs = []
    for _ in range(num_permutations):
        permuted_y = np.random.permutation(rank_y)
        permuted_cov_rank_xy = weighted_covariance(rank_x, permuted_y, normalized_weights)
        permuted_corr = permuted_cov_rank_xy / np.sqrt(var_rank_x * weighted_variance(permuted_y, normalized_weights))
        permuted_corrs.append(permuted_corr)

    permuted_corrs = np.array(permuted_corrs)
    p_value = np.sum(permuted_corrs >= weighted_rank_corr) / num_permutations

    return weighted_rank_corr, p_value

def autocorrelation(x, lag=1):
    """
    计算时间序列在指定滞后下的自相关系数。

    参数:
    x (array-like): 时间序列数据。
    lag (int): 滞后阶数，默认为 1。

    返回:
    float: 自相关系数。
    """
    n = len(x)
    if lag >= n:
        raise ValueError("Lag is too large for the length of the time series.")

    x_mean = np.mean(x)
    numerator = np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean))
    denominator = np.sum((x - x_mean)**2)

    return numerator / denominator