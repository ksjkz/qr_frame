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






class Get_data:
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
    
    def update_data(self, file_path:str='d2.csv',key:str='df',cul_adj_opt:str='qfq',save_path:str=''):

        
        today=datetime.now().strftime('%Y%m%d')#str like '20240830'
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        print(f'正在执行 更新A股日频数据到今天({today})')

        while True:
             try:
                  trade_days_list=self.get_trade_days(start_date= one_year_ago,end_date=today)
                  break
             except:
                  time.sleep(2)
        print('交易日列表获取完成')

        print(f"Loading data from {file_path}...")
        if re.match(r'.*\.h5$', file_path, re.IGNORECASE) is not None:
             df = pd.read_hdf(file_path, key=key)
        elif re.match(r'.*\.csv$', file_path, re.IGNORECASE) is not None :
       
             df=pd.read_csv(file_path)
             df.reset_index(inplace=True,drop=True)
             try:
                df.drop(columns=[ 'Unnamed: 0'], inplace=True)
             except:
                 pass

        df_trade_days_list=df['trade_date'].unique().astype(str)
        residue_date_list=[item for item in trade_days_list if item not in df_trade_days_list]#获取df到现在缺失的天
        print(f'{file_path}还要补全的天有{residue_date_list}')
        df.set_index(['ts_code', 'trade_date'], inplace=True)

        while True:
           try:
                 d1=self.get_day_data(trade_date= residue_date_list[0],if_get_adj_factor=True)
                 break
           except:
                time.sleep(2)
        print(f'{residue_date_list[0]}的数据已经获得')

        if len(residue_date_list)>1:
          print('现在获取剩下几天的数据')
          for i in tqdm(residue_date_list[1:],leave=True,initial=1,):
             while True:
                 try:
                      d2=self.get_day_data(trade_date= i,if_get_adj_factor=True)
                      d1=pd.concat([d1,d2],axis=0)
                      break
                 except :
                      time.sleep(2)
          print('剩下几天的数据已经获得')
        df=pd.concat([d1,df],axis=0)
        df=df.reset_index()

        if cul_adj_opt=='qfq':
            print('正在计算前复权 赋值到列如qfq_close')
            for _,group in df.groupby('ts_code'):
                df.loc[group.index,'qfq_open'] = group['open'] * group['adj_factor'] / group['adj_factor'].iloc[0]
                df.loc[group.index,'qfq_close'] = group['close'] * group['adj_factor'] / group['adj_factor'].iloc[0]
                df.loc[group.index,'qfq_high'] = group['high'] * group['adj_factor'] / group['adj_factor'].iloc[0]
                df.loc[group.index,'qfq_low'] = group['low'] * group['adj_factor'] / group['adj_factor'].iloc[0]
        elif cul_adj_opt=='hfq':
            print('正在计算后复权 赋值到列如hfq_close')
            for group in df.groupby('ts_code'):
                df.loc[group.index,'hfq_open'] = group['open'] * group['adj_factor'] / group['adj_factor'].iloc[-1]
                df.loc[group.index,'hfq_close'] = group['close'] * group['adj_factor'] / group['adj_factor'].iloc[-1]
                df.loc[group.index,'hfq_high'] = group['high'] * group['adj_factor'] / group['adj_factor'].iloc[-1]
                df.loc[group.index,'hfq_low'] = group['low'] * group['adj_factor'] / group['adj_factor'].iloc[-1]
        
        else :
            print(f'输入cul_adj_opt为{cul_adj_opt},不是qfq或者hfq,因此没有就adj_factor进行计算复权价格操作')



        if save_path=='':
            pass
        else:
            if re.match(r'.*\.h5$', file_path, re.IGNORECASE) is not None:
               df.to_hdf(save_path,key='df', mode='w')
               print(f'已经保存到{save_path}')
            elif re.match(r'.*\.csv$', file_path, re.IGNORECASE) is not None :
               df.to_csv(save_path)
               print(f'已经保存到{save_path}')
            else:
                print('输入的save_path不是h5或者csv')
                pass

        return df
    


        
                        
                 
            
        





        
             



    


        
 
        

    