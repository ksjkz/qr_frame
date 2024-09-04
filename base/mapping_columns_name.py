
"""
这个库本质就是个re映射库,用来统一公式和df的列名
"""
import re
#yd的意思是懿德财富
yd_mapping = {
    r'TICKER': 'ticker',
    r'T_DATE': 't_date',
    r'ADJ_D_OPEN': 'open_adj',
    r'ADJ_D_HIGH': 'high_adj',
    r'ADJ_D_LOW': 'low_adj',
    r'ADJ_D_CLOSE': 'close_adj',
    r'MCAP_IN_CIRCULATION': 'a_share_mcap_in_circulation',
    r'YUAN_VOLUME': 'dollar_volume',
    r'CLASS_NAME': 'class_name',
    r'PB_RATIO': 'pb_ratio',
    
    r'ONE_DAY_RETURN_PREV': 'ret_lag_close',
    r'D_OPEN_CHANGE': 'ret_lag_open',  #t日开盘相对于t-1日收盘的变化,adj过了
    r'D_HIGH_CHANGE': 'ret_lag_high',  #t日高相对于t-1日收盘的变化,adj过了
    r'D_LOW_CHANGE': 'ret_lag_low',     #t日低相对于t-1日收盘的变化,adj过了
    r'D_CLOSE_CHANGE': 'ret_lag_close',  
    r'D_VOLUME_CHANGE': 'dollar_volume_change',   # t日成交量相对于t-1日成交量变化
    r'D_PB_CHANGE': 'pb_ratio_change',   #pb_ratio t日相对于t-1日
    r'LOG_MCAP': 'log_mcap'    #np.log(a_share_mcap_in_circulation)
}


yd_mapping1 = {
   
    r'ret_lag_open': '(open_adj-close_adj/(1+ret_lag_close))/close_adj*(1+ret_lag_close)',  #t日开盘相对于t-1日收盘的变化,adj过了
    r'ret_lag_high': '(high_adj-close_adj/(1+ret_lag_close))/close_adj*(1+ret_lag_close)',  #t日高相对于t-1日收盘的变化,adj过了
    r'ret_lag_low': '(low_adj-close_adj/(1+ret_lag_close))/close_adj*(1+ret_lag_close)',     #t日低相对于t-1日收盘的变化,adj过了
  
    r'dollar_volume_change': 'ts_pct_change(dollar_volume,1)',   # t日成交量相对于t-1日成交量变化
    r'pb_ratio_change': 'ts_pct_change(pb_ratio,1)',   #pb_ratio t日相对于t-1日
    r'log_mcap': 'log(a_share_mcap_in_circulation)'    #np.log(a_share_mcap_in_circulation)
}


tushare_mapping = {
    r'TICKER': 'ts_code',
    r'T_DATE': 'trade_date',
    r'ADJ_D_OPEN': 'open_adj',
    r'ADJ_D_HIGH': 'high_adj',
    r'ADJ_D_LOW': 'low_adj',
    r'ADJ_D_CLOSE': 'close_adj',
    r'MCAP_IN_CIRCULATION': 'circ_mv*10000',
    r'YUAN_VOLUME': 'amount*10000',
    r'PB_RATIO': 'pb',
    
    r'ONE_DAY_RETURN_PREV': '(close_adj-adj_pre_close)/adj_pre_close',
    r'D_OPEN_CHANGE': '(open_adj-adj_pre_close)/adj_pre_close',  #t日开盘相对于t-1日收盘的变化,adj过了
    r'D_HIGH_CHANGE': '(high_adj-adj_pre_close)/adj_pre_close',  #t日高相对于t-1日收盘的变化,adj过了
    r'D_LOW_CHANGE': '(low_adj-adj_pre_close)/adj_pre_close',     #t日低相对于t-1日收盘的变化,adj过了
    r'D_CLOSE_CHANGE': '(close_adj-adj_pre_close)/adj_pre_close',  
    r'D_VOLUME_CHANGE': 'ts_pct_change(amount,1)',   # t日成交量相对于t-1日成交量变化
    r'D_PB_CHANGE': 'ts_pct_change(pb,1)',   #pb_ratio t日相对于t-1日
    r'LOG_MCAP': 'log(circ_mv*10000)'    #np.log(a_share_mcap_in_circulation)
}

tushare_mapping1 = {

    r'open_adj': 'open*adj_factor/last_adj',
    r'high_adj': 'high*adj_factor/last_adj',
    r'low_adj': 'low*adj_factor/last_adj',
    r'close_adj': 'close*adj_factor/last_adj',
    r'adj_pre_close': 'pre_close*adj_factor/last_adj',
    
}


def trans_(input:str|list[str],mapping_dict:dict)->str|list[str]:
     '''
     这个函数根据mapping_dict通过正则表达式替换公式统一公式和df的列名
     输入list[str]或者str
     '''
     if isinstance(input, list):
            a=[]
            for i in input:
                   a.append(trans_(i,mapping_dict))
            return a
    
     elif isinstance(input, str):
        for old, new in mapping_dict.items():
              input = re.sub(old, new, input)
        return  input
     else:
       raise TypeError("变量必须是 list[str] 或者 str")
     
def trans_old2_yd_data(input:str)->str:
     '''
     懿德财富数据集的映射
     '''
     a=trans_(input,yd_mapping)
     a=trans_(a,yd_mapping1)
     return a


def trans_old2_tushare_data(input:str)->str:
     '''
     tushare数据集的映射
     '''
     a=trans_(input,tushare_mapping)
     a=trans_(a,tushare_mapping1)
     return a
     



       
    
             

    