import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType

class Trading:
    """
    初始化参数
    df initial_capital transaction_cost time_name
    方法
    find_value(trade_date, ticker,column)

    get_next_day(trade_date)

    get_prev_day(trade_date)

    trans_time(trade_date)

    buy(ticker,trade_date,buy_price,investment) 

    sell(ticker,quantity,trade_date,sell_price)

    merge_holdings()

    get_total_capital(trade_date,is_print=True)
    
    total_capital_line_plot()
    """
    def __init__(self,
                 df: pd.DataFrame,  #df过一下load_df
                 initial_capital=150_000,                      #初始资金
                 transaction_cost=0.000085,                    #券商佣金,印花税另加的(印花卖出万5,过户费沪市双边万1)
                 time_name:str='trade_date',
                 ):
                
        self.df = df   
        self.time_name = time_name                      
        self.trade_dates_list = sorted(self.df[time_name].unique()) #交易日期的list
        self.initial_capital = initial_capital  
        self.transaction_cost = transaction_cost          
        self.cash = initial_capital                      # 当前现金 
        self.holdings_for_sell = []                      # 可被出售的持仓，格式为 [{'ticker': '000001.SZ', 'quantity': 100, 'average_cost': 10,}] 
        self.holdings_not_for_sell = []                  # 不可被出售的持仓，为当天买入,trade_date晚上会被更新到self.holdings_for_sell,格式同上
        self.holdings_list = []                          #每天结束记录当天晚上持仓
        self.orders = []                                 # 交易记录，格式为 [(交易日期, SELL, 股票代码, 交易数量, 卖出价格, 交易收益),
        self.total_capital={}                            # 总资产 格式为 {交易日期: 总资产} 当日收盘后
       
    def find_value(self, trade_date, ticker,column:str)->int|float|bool:
        try:
           value = self.df[(self.df[self.time_name] == trade_date) & (self.df[self.ticker_name] == ticker)][column].values[0]
        except Exception as e:
           print(e)
           print('找不到对应值',trade_date, ticker, column)
           value=False
        return value
        
    def get_next_day(self, trade_date):
        return self.trade_dates_list[self.trade_dates_list.index(trade_date) + 1]
    def get_prev_day(self, trade_date):
        return self.trade_dates_list[self.trade_dates_list.index(trade_date) - 1]
    def trans_time(self,trade_date:str):
           return datetime.strptime(trade_date, "%Y-%m-%d").date()
    def buy(self,ticker:str,trade_date:datetime.date,buy_price:float,investment:float)->None:  #在trade_date盘中买入,buy_price为买入价格(原始价格，investment为投入金额
         if isinstance(trade_date, str):
            trade_date = self.trans_time(trade_date)
         
         if ticker.startswith(('60')): #沪市有过户费
                            buy_tax=1 + self.transaction_cost+0.0001
         else:
                            buy_tax=1 + self.transaction_cost
         quantity = investment // (buy_price * 100 * buy_tax)
         cost = round(quantity * buy_price * buy_tax * 100,2)
         if quantity > 0:
                            self.cash = self.cash-cost
                            buy_price1=cost/(quantity*100)   #算交易费用复合成本 quantity是股票手数
                            self.holdings_not_for_sell.append({'ticker': ticker, 'quantity': quantity, 'average_cost': buy_price1})
                            self.orders.append({'ticker': ticker, 'type':'BUY', 'date': trade_date,'quantity': quantity, 'price': buy_price1, 'total_money':cost})
         elif quantity == 0:
               pass
         else:
               raise ValueError('请检查investment,买入数量不能为负数')
               

    def sell(self,ticker:str,quantity:int,trade_date:datetime.date,sell_price:float)->None: #在trade_date盘中卖出
         current_holding = [item for item in self.holdings_for_sell if item.get('ticker') == ticker][0]
         if 0>=quantity:
               raise ValueError('卖出数量不能为负数和0')
               
         if ticker.startswith(('60')): #沪市有过户费
                            sell_tax=1 - self.transaction_cost-0.0006
         else:
                            sell_tax=1 - self.transaction_cost-0.0005
         collection=round(sell_price *quantity*sell_tax *100,2)
         new_quantity = current_holding.get('quantity') - quantity
         sell_price1=collection/(quantity*100)
         if new_quantity > 0:
                            self.cash = self.cash+collection
                            current_holding['average_cost'] = (current_holding.get('average_cost') * current_holding.get('quantity') - collection) / new_quantity
                            current_holding['quantity'] = new_quantity
                            self.orders.append({'ticker': ticker, 'type':'SELL','date': trade_date,'quantity': quantity, 'price': sell_price1,'total_money':collection})
         elif new_quantity == 0:
                            self.cash = self.cash+collection
                            self.holdings_for_sell.remove(current_holding)
                            self.orders.append({'ticker': ticker, 'type':'SELL','date': trade_date,'quantity': quantity, 'price': sell_price1,'total_money':collection})
         else:
                            raise ValueError('卖出数量大于持有数量')
                            

    def merge_holdings(self):  #合并持仓
      if self.holdings_not_for_sell==[]:
        return
      for new_holding in self.holdings_not_for_sell:
        ticker = new_holding['ticker']
        quantity = new_holding['quantity']
        average_cost = new_holding['average_cost']
        # 查找是否在 holdings_for_sell 中存在相同的 ticker
        existing_holding = next((item for item in self.holdings_for_sell if item.get('ticker') == ticker), None)
        if existing_holding:
            # 更新现有持仓：合并数量并重新计算平均成本
            total_quantity = existing_holding['quantity'] + quantity
            new_average_cost = (
                (existing_holding['average_cost'] * existing_holding['quantity'] + average_cost * quantity) 
                / total_quantity
            )
            existing_holding['quantity'] = total_quantity
            existing_holding['average_cost'] = new_average_cost
        else:
            # 如果 holdings_for_sell 中没有该 ticker，直接添加
            self.holdings_for_sell.append(new_holding)
            # 清空 holdings_not_for_sell
        self.holdings_not_for_sell = []

    
    def get_total_capital(self,trade_date,is_print=True)->float: #在merge_holdings后，计算总资产
        total_capital=0
        for item in self.holdings_for_sell:
            total_capital += item.get('quantity') * self.find_value(trade_date=trade_date,ticker=item.get('ticker'),column=self.close_name) * 100
        total_capital=round(total_capital+self.cash,2)
        if is_print:
            print(trade_date,'总资产',total_capital)
        self.total_capital[trade_date]=total_capital
        return total_capital
    
    def total_capital_line_plot(self,):
         y_data  = list(self.total_capital.values())
         x_data = list(self.total_capital.keys())
         line = (
                          Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))  
                         .add_xaxis(xaxis_data=x_data)
                               .add_yaxis(
                              series_name="账户总资产",  
                              y_axis=y_data,         
                              is_smooth=True,  
                              label_opts=opts.LabelOpts(is_show=False)      
                         )
                            .set_global_opts(
                            title_opts=opts.TitleOpts(title="总资产随时间变化", subtitle=""),
                                   xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
                                    yaxis_opts=opts.AxisOpts(type_="value"),
                                        )
                                       )
         return line.render_notebook()