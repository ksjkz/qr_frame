
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



class Backtesting:
    """
        类名：Backtesting
        单因子/模型回测框架
------------------输入参数
                        ---df 包含所有信息的完整的df(注意初始化的时候会自动将df进行dropna)
                        ---factor_name 因子名称 为df的某一列的列名
                        ---r_name:匹配return的列名
                        ---t_name:匹配time的列名(日期,月之类的)
                        ---opt 0 四分法去极值加标准化
                               1 仅去极值
                               2 仅标准化
                               其他 不做任何处理

                        
                        通过__init__相当于得到数据清洗以后的df    
                        这时候的三个属性
                        self.df
                        self.r_name
                        self.t_name

--------------------方法section_cul:截面回归得到ic序列之类 详情见函数解析
                    得到section_cul_df  记录所有常见属性的时间序列

--------------------方法:time_cul:时序,计算section_cul_df 中各种指标序列的mean ir
                    得到time_cul_dict 记录所有常见属性的时间序列  

--------------------方法:rolling_cul:滚动累加,计算section_cul_df 中各种指标序列的mean ir
                    添加到section_cul_df 的列中
    """
   
    def __init__(self,df:pd.DataFrame,factor_name:str='',opt:int=0,r_name:str='',t_name:str='',ticker_name:str=''):
                 
        if not t_name   in df.columns:
                   raise ValueError(f"t_name '{t_name}' is either empty or not found in DataFrame columns.")
        if not r_name   in df.columns:
                   raise ValueError(f"r_name '{r_name}' is either empty or not found in DataFrame columns.")
        self.df=df.dropna()
        self.r_name=r_name
        self.t_name=t_name
        self.ticker_name=ticker_name
        if factor_name=='' or (not (factor_name in self.df.columns)):
            raise ValueError('factor_name is wrong')
        else:
            self.factor_name=factor_name
        

        match opt:
          case 0:
                lenth1=self.df.shape[0]
                for _,group in tqdm(self.df.groupby(t_name),leave=False,desc='正在清洗df'):
                                
                                Q1 = group[self.factor_name].quantile(0.25)
                                Q3 = group[self.factor_name].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                group.loc[
                                     (group[self.factor_name] < lower_bound) | (group[self.factor_name] > upper_bound),
                                     [self.factor_name, r_name]
                                         ] = None
                                a=group[self.factor_name].mean()
                                b=group[self.factor_name].std()
                                a1=group[r_name].mean()
                                b1=group[r_name].std()
                                self.df.loc[group.index, self.factor_name] =group[self.factor_name].apply(lambda x: (x - a) / b)
                                self.df.loc[group.index, r_name]=group[r_name].apply(lambda x: (x - a1) / b1)

                self.df=self.df.dropna()
                lenth2=self.df.shape[0]
                logger.info(f'已经dropna并将{self.factor_name}和{r_name}标准化并按四分法去极值,去极值前有{lenth1}条数据，去极值后有{lenth2}条数据')
                
          case 1:
             lenth1=self.df.shape[0]
             for _,group in tqdm(self.df.groupby(t_name),leave=False,desc='正在清洗df'):
                 Q1 = group[self.factor_name].quantile(0.25)
                 Q3 = group[self.factor_name].quantile(0.75)
                 IQR = Q3 - Q1
                 lower_bound = Q1 - 1.5 * IQR
                 upper_bound = Q3 + 1.5 * IQR
                 self.df.loc[group.index, self.factor_name]=group[self.factor_name].apply(lambda x: None if x < lower_bound or x >upper_bound  else x)

             self.df=self.df.dropna()
             lenth2=self.df.shape[0]
             logger.info(f'已经dropna并按四分法去极值,去极值前有{lenth1}条数据，去极值后有{lenth2}条数据')

          case 2:
            for _,group in tqdm(self.df.groupby(t_name),leave=False,desc='正在清洗df'):
                a=group[self.factor_name].mean()
                b=group[self.factor_name].std()
                a1=group[r_name].mean()
                b1=group[r_name].std()
                self.df.loc[group.index, self.factor_name] =group[self.factor_name].apply(lambda x: (x - a) / b)
                self.df.loc[group.index, r_name]=group[r_name].apply(lambda x: (x - a1) / b1)
            logger.info(f'已经dropna并将{self.factor_name}和{r_name}标准化')
        
          case _:
                    logger.info(f'仅仅dropna,没做任何处理')
                    pass
        logger.info(f'已完成backtest初始化,df的shape为{self.df.shape}')
        
        
    def section_cul(self,opt:int|str=0):
        """
        opt:
            0:不加权
            z:按z加权
        """
        grouped_df=self.df.groupby(self.t_name)
        backtest_list=[]    
        match opt:
            case 0:
                for name,group in tqdm(grouped_df,leave=True,desc='正在回测,不加权'):
                         rank_ic, p_rank_ic = spearmanr(group[self.factor_name],group[self.r_name])
                         ic, p_ic = pearsonr(group[self.factor_name],group[self.r_name])
                         i={}
                         i['time']=name
                         i['rank_ic']=rank_ic
                         i['ic']=ic
                         i['p_rank_ic']=p_rank_ic
                         i['p_ic']=p_ic
                         backtest_list.append(i)
            case z:
                    if not(z in self.df.columns):
                        raise ValueError(f'{z} NOT in self.df.columns')
                    for name,group in tqdm(grouped_df,leave=True,desc=f'正在回测,按{z}加权'):
                           WC = WeightedCorr(xyw=group[[self.factor_name,self.r_name,z]])
                           ic=WC(method="pearson")
                           rho=WC( method="spearman")
                           i={}
                           i['time']=name
                           i['ic']=ic
                           i['rank_ic']=rho
                           backtest_list.append(i)

        section_cul_df = pd.DataFrame(backtest_list)
        setattr(self, 'section_cul_df', section_cul_df)
        logger.info(f'已经成功进行section_cul,保存在属性section_cul_df里,计算选项opt为{opt}')
        return section_cul_df

    def get_basic_info(self,):
            """
            功能描述:
            计算section_cul_df的每一列的mean
            rankic的icir.
            同时记录period即数据开始结束时间
            """
            if not hasattr(self, 'section_cul_df'):
                  self.section_cul()
            columns_list = self.section_cul_df.columns.to_list()
            columns_list.remove('time')
            aa={}
            for columns_name in columns_list:
                aa[f'{columns_name}_mean']=self.section_cul_df[columns_name].mean()
                
                
            icir=self.section_cul_df['rank_ic'].mean()/self.section_cul_df['rank_ic'].std()*np.sqrt(self.section_cul_df['rank_ic'].shape[0]-1)
            aa['icir']=icir
            aa['open_time']=self.section_cul_df['time'][0]
            aa['close_time']=self.section_cul_df["time"].iloc[-1]
            setattr(self, 'time_cul_dict', aa)
            logger.info(f'已经成功进行time_cul,保存在属性time_cul_dict里')
            return aa

    def get_cum_info(self,):
          '''
          section_cul_df 列计算累加值
          '''
          if not hasattr(self, 'section_cul_df'):
                  self.section_cul()
          columns_list = self.section_cul_df.columns.to_list()
          columns_list.remove('time')
          for columns_name in columns_list:
               self.section_cul_df[f'{columns_name}_cum']=self.section_cul_df[columns_name].cumsum()
          logger.info(f'已经成功进行rolling_cul,累加列保存在属性section_cul_df里')
          return self.section_cul_df
    def run(self,opt:int|str=0,if_get_autocorr=False,autocorr_n:int=10):
        '''
        汇总执行其他方法
        opt:0代表不加权,除0以外代表加权,opt为权重列名
        if_get_autocorr:是否计算自相关系数 可以得到rankic的自相关系数 和factor groupby('ticker')以后的自相关系数
        '''
        self.section_cul(opt=opt)
        self.get_basic_info()
        self.get_cum_info()
        if if_get_autocorr:
           self.get_autocorr(autocorr_n)
           self.get_grouped_acf(autocorr_n)
        
        
    def get_autocorr(self,n:int=10,):
          if not hasattr(self, 'section_cul_df'):
                  self.section_cul()
          y1=self.section_cul_df['rank_ic'].to_list()
          
          autocorr = acf(y1, nlags=n)[1:]
          setattr(self, 'autocorr_rankic_day', autocorr)
          logger.info(f'已经成功对rankic_mean进行acf计算,保存在属性autocorr_rankic_mean_day里,为list,保存滞后1到{n}期的autocorr')
          return autocorr

    def get_grouped_acf(self,n:int=10,):
         grouped = self.df.groupby(self.ticker_name)
         acf_results = pd.DataFrame()
         for name, group in grouped:
                       acf_values = acf(group[self.factor_name].to_list(), nlags=n, fft=False)
                       acf_df = pd.DataFrame({
                                  'group_col': [name] * len(acf_values),
                                  'lag': np.arange(len(acf_values)),
                                   'acf': acf_values
                                   })
                       acf_results = pd.concat([acf_results, acf_df], ignore_index=True)
         setattr(self, 'autocorr_by_ticker_df',  acf_results)
         logger.info(f'已经成功groupby ticker对{self.factor_name}进行acf计算,保存在属性autocorr_by_ticker_df里,保存滞后1到{n}期的autocorr')
         mmm=self.autocorr_by_ticker_df.groupby('lag')['acf'].mean().reset_index()
         setattr(self, 'autocorr_by_ticker_mean_df', mmm)
         logger.info(f'已经成功对autocorr_by_ticker_df groupby(lag)进行mean,保存在属性autocorr_by_ticker_mean_df里')

         return acf_results,mmm
          
              
                  


def get_decile_return(df: pd.DataFrame, by: str='', n: int = 10,w_opt=0,r_opt:int=0,t_name:str='',r_name:str='') -> pd.DataFrame|list[dict]:
         """
        功能描述:
         按截面按因子值分为n组，计算每组加权收益率

        参数:
          df: 包含时间,return和因子值的df
          by: 因子值的列名
          n: 分组数
          w_opt: 0不加权，其他请输入权重的列名
          r_opt 0返回df,1返回dict

        返回值:
             函数返回值的描述包含的key:  time	decile	weighted_return
         
         """
         df=df.copy()
         if not t_name  in df.columns:
                   raise ValueError(f"t_name '{t_name}' is either empty or not found in DataFrame columns.")
         if not r_name  in df.columns:
                   raise ValueError(f"r_name '{r_name}' is either empty or not found in DataFrame columns.")
         if by not in df.columns:
                   raise ValueError(f"by '{by}' is either empty or not found in DataFrame columns.")
         

         results=[]
         for name,group in tqdm(df.groupby(t_name),desc='正在分组回测计算decile_return'):
             group = group.sort_values(by=by)
             group['decile'] = pd.qcut(group[by], n, labels=False,duplicates='drop')
             desc=''
             for decile in range(n):
                          decile_group = group[group['decile'] == decile]
                          if decile_group.empty:#如果分组不存在,将weighted_return置0,避免报错
                                 results.append({'time': name, 'decile': decile, 'weighted_return': 0})
                                 print(f'第{name} 分组{decile}不存在')
                                 continue
                          match w_opt:
                                 case 0:
                                         weighted_return=decile_group[r_name].mean()
                                         desc='不加权'
                                 
                                 case z:
                                        if not(z in df.columns):
                                                   raise ValueError(f'{z} NOT in self.df.columns')
                            
                                        weighted_return = np.average(decile_group[r_name], weights=decile_group[z])
                                        desc=f'按{z}加权'

                          results.append({'time': name, 'decile': decile, 'average_return': weighted_return})

         match r_opt:
                case 0:
                       result_df = pd.DataFrame(results)
                       logger.info(f'分组计算return完成,{desc},返回df')
                       return result_df
                case 1:
                       logger.info(f'分组计算return完成,{desc},返回list[dict]')
                       return results
               
                case _:
                       raise ValueError('r_opt参数只能为0或1')
                

class Decile:
    """
    分组计算每组收益率
     df: 包含时间,return和因子值的df
     f_name: 因子值的列名
     n: 分组数
     w_opt: 0不加权，其他请输入权重的列名
     t_name: 时间列名
     r_name: 收益率列名
     if_run: 是否自动运行
    """
    def __init__(self,df: pd.DataFrame, f_name: str='', n:int = 10,w_opt:int|str =0,t_name:str='',r_name:str='',if_run=True) -> pd.DataFrame:
        self.df=get_decile_return(df=df, by=f_name, n=n,w_opt=w_opt,r_opt=0,t_name=t_name,r_name=r_name)
        self.df['time'] = pd.to_datetime(self.df['time'])
        if if_run:
            self.run()

    def get_mean(self):
        setattr(self, 'mean_df', self.df.groupby('decile')['average_return'].mean())
        return self.mean_df
    def get_std(self):
        setattr(self, 'std_df', self.df.groupby('decile')['average_return'].std())
        return self.std_df
    def get_cumprod(self):
        for _,group in self.df.groupby('decile'):
                  group['weighted_return']=group['average_return']+1
                  self.df.loc[group.index,'average_return_cumprod']=group['weighted_return'].cumprod()
        return self.df
    def get_cumsum(self):
        for _,group in self.df.groupby('decile'):
                  self.df.loc[group.index,'average_return_cumsum']=group['weighted_return'].cumsum()
        return self.df
    def run(self):
           self.get_mean()
           self.get_cumprod()
           return self.df,self.mean_df
    


    
    
                       
