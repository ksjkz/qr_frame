
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


params = {
    '_time': 'T_DATE',
    '_ticker': 'TICKER',
    '_class_name': 'CLASS_NAME',
    '_return': 'ONE_DAY_RETURN',  
    '_mcap_in_circulation': 'MCAP_IN_CIRCULATION',
    '_log_mcap_in_circulation':'LOG_MCAP'
}

class Backtesting:
    """
        类名：Backtesting
------------------输入参数
                        ---df 包含所有信息的完整的df(注意初始化的时候会自动将df进行dropna)
                        ---factor_name 因子名称 为df的某一列的列名
                        ---r_name:匹配return的列名
                        ---t_name:匹配time的列名(日期,月之类的)
                        ---opt (0,_,_) 四分法去极值加标准化
                               (_,0,_) 仅去极值
                               (_,1,_) 仅标准化
                               (_,_,_) 不做任何处理

                        
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
   
    def __init__(self,df:pd.DataFrame,factor_name:str='',opt=(0,0,0),r_name:str=params['_return'],t_name:str=params['_time'],if_return=False
                 ):
        self.df=df.dropna()
        self.r_name=r_name
        self.t_name=t_name
        if factor_name=='' or (not (factor_name in self.df.columns)):
            raise ValueError('factor_name is wrong')
        else:
            self.factor_name=factor_name
        

        match opt:
          case (0,_,_):
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
                logger.info(f'已经将{self.factor_name}和{r_name}标准化并按四分法去极值,去极值前有{lenth1}条数据，去极值后有{lenth2}条数据')
                
          case (_,0,_):
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
             logger.info(f'已经按四分法去极值,去极值前有{lenth1}条数据，去极值后有{lenth2}条数据')

          case(_,1,_):
            for _,group in tqdm(self.df.groupby(t_name),leave=False,desc='正在清洗df'):
                a=group[self.factor_name].mean()
                b=group[self.factor_name].std()
                a1=group[r_name].mean()
                b1=group[r_name].std()
                self.df.loc[group.index, self.factor_name] =group[self.factor_name].apply(lambda x: (x - a) / b)
                self.df.loc[group.index, r_name]=group[r_name].apply(lambda x: (x - a1) / b1)
            logger.info(f'已经将{self.factor_name}和{r_name}标准化')
        
          case(_,_,_):
                    pass
        logger.info(f'已完成backtest初始化,df的shape为{self.df.shape}')
        if if_return:
          return self.df
        
    def section_cul(self,opt:tuple=(1,0,0),if_return=False):
        """

        opt:x,y,z:默认值,计算ic和rankic
        x:代表计算相关系数的时候加不加权 0代表不加权,除0以外代表加权
        y:0代表用流通市值加权,1代表用log流通市值加权
        z:在y!=0或1的基础上,z为加权列的列名,类型为 str
    
        """
        grouped_df=self.df.groupby(self.t_name)
        backtest_list=[]    
        match opt:
            case (0,_,_):
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

            case (_,0,_):
                weight=params['_mcap_in_circulation']
                if not(weight in self.df.columns):
                        raise ValueError(f'{weight} NOT in self.df.columns')
                weight=params['_mcap_in_circulation']
                for name,group in tqdm(grouped_df,leave=True,desc='正在回测,按流通市值加权'):
                           WC = WeightedCorr(xyw=group[[self.factor_name,self.r_name,weight]])
                           ic=WC(method="pearson")
                           rho=WC( method="spearman")
                           i={}
                           i['time']=name
                           i['ic']=ic
                           i['rank_ic']=rho
                           backtest_list.append(i)
            case (_,1,_):
                weight=params['_log_mcap_in_circulation']
                if not(weight in self.df.columns):
                        raise ValueError(f'{weight} NOT in self.df.columns')
                for name,group in tqdm(grouped_df,leave=True,desc='正在回测,按log流通市值加权'):
                           WC = WeightedCorr(xyw=group[[self.factor_name,self.r_name,weight]])
                           ic=WC(method="pearson")
                           rho=WC( method="spearman")
                           i={}
                           i['time']=name
                           i['ic']=ic
                           i['rank_ic']=rho
                           backtest_list.append(i)
            case (_,_,z):
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
        if if_return:
            return section_cul_df

    def get_basic_info(self,if_return=False):
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
            if if_return:
                return aa

    def get_cum_info(self,if_return=False):
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
          if if_return:
              return self.section_cul_df
    

    def get_autocorr(self,n:int=10,if_return=False):
          if not hasattr(self, 'section_cul_df'):
                  self.section_cul()
          y1=self.section_cul_df['rank_ic'].to_list()
          
          autocorr = acf(y1, nlags=n)[1:]
          setattr(self, 'autocorr', autocorr)
          logger.info(f'已经成功进行acf计算,保存在属性autocorr里,为list,保存滞后1到{n}期的autocorr')
          if if_return:
              return autocorr
          
              
                  


def decile_return(df: pd.DataFrame, by: str='', n: int = 10,opt:tuple =(1,0,0,0),t_name:str=params['_time'],r_name:str=params['_return']) -> pd.DataFrame|list[dict]:
         """
        功能描述:
         按截面按因子值分为n组，计算每组加权收益率

        参数:
          df: 包含时间,return和因子值的df
          by: 因子值的列名
          n: 分组数
          opt 选项 (is-weight,weight_opt,weight_by,r_opt)
          is-weight 0不加权,其他加权
          weight_opt 0:市值,1:log市值, _:自定义加权
          weight_by:自定义加权的列名
          r_opt 0返回df,1返回dict

        返回值:
             函数返回值的描述包含的key:  time	decile	weighted_return
         
         """
         if by=='':
             raise ValueError('by参数不能为空')

         results=[]
         w_opt=opt[:-1]
         r_opt=opt[-1]
         for name,group in tqdm(df.groupby(t_name),desc='正在分组回测计算decile_return'):
             group = group.sort_values(by=by)
             group['decile'] = pd.qcut(group[by], n, labels=False,duplicates='drop')
             desc=''
             for decile in range(n):
                          decile_group = group[group['decile'] == decile]
                          if decile_group.empty:#如果分组不存在,将weighted_return置0,避免报错
                                 results.append({'time': name, 'decile': decile, 'weighted_return': 0})
                                 continue
                          match w_opt:
                                 case (0,_,_):
                                         weighted_return=decile_group[r_name].mean()
                                         desc='不加权'
                                 case (_,0,_):
                                     weighted_return = np.average(decile_group[r_name], weights=decile_group[params['_mcap_in_circulation']])
                                     desc='按市值加权'
                                 case (_,1,_):
                                        weighted_return = np.average(decile_group[r_name], weights=decile_group[params['_log_mcap_in_circulation']])
                                        desc='log市值加权'
                                 case (_,_,z):
                                        if not(z in df.columns):
                                                   raise ValueError(f'{z} NOT in self.df.columns')
                                        
                                        weighted_return = np.average(decile_group[r_name], weights=decile_group[z])
                                        desc=f'按{z}加权'

                                 
                          results.append({'time': name, 'decile': decile, 'weighted_return': weighted_return})

        
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
                       
