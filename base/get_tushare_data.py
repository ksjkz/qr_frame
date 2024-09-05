'''
这个文件,用于通过tushare获取A股的日行情数据

'''
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import yaml
import requests as req
import re
from tqdm import tqdm
import time
from base.my_os_func import get_all_dir,read_csv_from_paths_to_one_df
from base.formulate_coding_AST import load_df





class Get_tushare_data:
    def __init__(self):
        with open("config/tushare_api.yaml", "r", encoding="utf-8") as file:
          config = yaml.safe_load(file)
        self.url = config['tushare_api']['url']
        self.token = config['tushare_api']['token']

    def get_token_validity_period(self):
        """
         获取token的有效期
         """
        url = self.url + '/TOKEN'
        data = {'TOKEN': self.token}
        response = req.post(url, data=data)
        return response.json()

    def get_trade_days(self,start_date:str='20240810',end_date:str='20240830'):
        """
          获取包含start_date和end_date之间的所有交易日的列表
          :param start_date: 开始日期，格式为YYYYMMDD 如'20200311'
          返回 list[str]
        """
        url = self.url + '/trade_cal'
        data = {'TOKEN':self.token,'start_date':start_date,'end_date':end_date,'exchange':'SSE'}
        trade_cal_df = pd.DataFrame(req.post(url, data=data).json())
        trade_days_df = trade_cal_df[trade_cal_df['is_open'] == 1]# 过滤出交易日
        trade_days_list = trade_days_df['cal_date'].tolist()# 提取交易日期列表

        return trade_days_list

    def get_day_data(self,trade_date:str='20240812',if_get_adj_factor=True):
        """
        获取某一天的所有A股的日行情数据
        """
        url1 = self.url + '/daily'
        url2 = self.url + '/daily_basic'
        data1 = {'TOKEN':self.token,'trade_date':trade_date,'adj':'qfq'}
        data2 = {'TOKEN':self.token,'trade_date':trade_date,'adj':'qfq','fields':'ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv'}
        daily_df = pd.DataFrame(req.post(url1, data=data1).json())
        daily_basic_df = pd.DataFrame(req.post(url2, data=data2).json())
        daily_basic_df = daily_basic_df.drop(columns=['close'])
        merged_df = pd.merge(daily_df, daily_basic_df, on=['ts_code', 'trade_date'], how='left')
        stock_list=merged_df['ts_code'].unique().tolist()

        if if_get_adj_factor:
             merged_df.set_index(['ts_code', 'trade_date'], inplace=True)
             merged_df['adj_factor']=.0
             def chunk_list(lst, chunk_size):
                    for i in range(0, len(lst), chunk_size):
                         if i + chunk_size > len(lst):
                             yield lst[i:]
                         else:
                             yield lst[i:i + chunk_size]
             for  chunk in chunk_list(stock_list, 1000):#数据量太大服务器不返回
                 while True:
                     try:
                         adj_factor_df = self.get_adj_factor(start_date=trade_date,end_date=trade_date,ts_code=','.join(chunk))
                         #adj_factor_df['trade_date']=adj_factor_df['trade_date'].dt.strftime('%Y%m%d').astype('int64')
                         adj_factor_df.set_index(['ts_code', 'trade_date'],inplace=True)
                         merged_df.update(adj_factor_df)
                         break
                     except:
                         pass
        merged_df = merged_df.reset_index()

        return merged_df

    def get_adj_factor(self,ts_code:str='000001.SZ',start_date:str='20230810',end_date:str='20230830'):
        """
        返回一只股票在某一段时间的adj_factor
        注意 也可以同时返回多支股票 stock_list = ['000001.SZ', '600519.SH', '000002.SZ']
                                  ts_code=','.join(stock_list)

        """
        url = self.url + '/adj_factor'
        data = {'TOKEN':self.token,'ts_code':ts_code,'start_date':start_date,'end_date':end_date,}
        start_date=pd.to_datetime(start_date)
        end_date=pd.to_datetime(end_date)
        adj_factor_df = pd.DataFrame(req.post(url, data=data).json())
        adj_factor_df['trade_date'] = pd.to_datetime(adj_factor_df['trade_date'])
        filtered_df = adj_factor_df[(adj_factor_df['trade_date'] >= start_date) & (adj_factor_df['trade_date'] <= end_date)]
        return filtered_df
    
def tushare_date_loader(dir_path:str='tushare')->pd.DataFrame:
    a=get_all_dir(dir_path)
    df=read_csv_from_paths_to_one_df(a)
    df=load_df(df,groupby_column='ts_code',sort_column='trade_date')
    return df



        
       


        
                        
                 
            
        





        
             



    


        
 
        

    