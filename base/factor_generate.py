import yaml
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict
from base.memory import *
import random

def generate_Initial_Factor(batch_name:str='D',factor_num:int=4,complexity:str='complex',is_show=False,save_opt=False) -> List[Dict[str, str]]:
  '''
           factor_Generating_Initial     用于随机生产初始因子
                            --------batch_name: 批次名称 生产因子的批次名称
                            --------is_show: 是否直接打印生成的因子
                            --------save_opt: 是否保存生成的因子到json文件

返回,形如
{
 'Factor Name': 'FM_VOLUME_ADJUSTED_VOLATILITY',
 'Factor Formula': "div(ts_std(mul(YUAN_VOLUME,sqrt(sub('ADJ_D_HIGH', 'ADJ_D_LOW'))),20),ts_mean('YUAN_VOLUME',20))",
 'Explanation': 'This factor measures the volatility of a stock, adjusted for the trading volume. It calculates the standard deviation of the trading volume multiplied by the square root of the difference between the highest and lowest prices, divided by the mean trading volume over a 20-day period. A higher value indicates a more volatile stock.'}

  '''
  i:int=0
  while(1):
     i=i+1
     if i>4:
         print('-------------------------因子产生失败-------------------')
         return False
     llm=open_New_LLM()
     # 读取 YAML 文件
     with open("config/params_for_generate.yaml", "r", encoding="utf-8") as file:
          config2= yaml.safe_load(file)
     input_params = config2["input_params"]
     input_params['batch_name'] = batch_name
     input_params['factor_num']=factor_num
     input_params['complexity'] = complexity
     template1 = config2["template"]
     json_name=batch_name +'_factors.json'
  
     prompt_template = PromptTemplate(
     template=template1,
       #input_variables=["basic_factors", "factor_terms", "factor_term_names", "num_factors", "num_formulas", "max_operation_length", "min_operation_length", "max_basic_factor_length","max_basic_factor_length"]
            )
     
     chain = LLMChain(llm=llm, prompt=prompt_template)
     output = chain.run(input_params)
     if is_show:
         print('生成的因子为:{}'.format(output))
     try:
         data_list = json.loads(output)
         if save_opt:
            data={
                 'input_params':input_params,
                  "factor_list":data_list
               }
     # 将数据写入JSON文件
            with open( json_name, 'w') as file:
                    json.dump(data, file, indent=4)
                    print("Data has been saved to {}".format(json_name))
         print('----------------------因子已经成功生成-----------------\n')
         return data_list
     except:
         print('-------------- -因子产生失败，正在尝试重新生成--------')
         continue
     

def generate_Improve_Factor(input:list[dict],factor_num:int=4,complexity:str='very_complex',batch_name:str='AA',is_show=False)->list[dict]:
          
      '''
      generate_Improve_Factor
      根据输入因子和评价指标生产改进后的因子
      评价指标 IC_mean,rank_ic_mean,icir

      输入输出因子形式:同上

      '''
      iii:int=0
      while(1):
             iii=iii+1
             if iii>3:
                 print('-------------------------改进因子失败-------------------')
                 return False
             
             llm=open_New_LLM()
             # 读取 YAML 文件
             with open("config/params_for_generate.yaml", "r", encoding="utf-8") as file:
                     config2= yaml.safe_load(file)
             input_params = config2["input_params"]
             template1 = config2["template1"]
             input_params['batch_name'] = batch_name
             input_params['factor_example']=input
             input_params['factor_num']=factor_num
             input_params['complexity'] = complexity
             prompt_template = PromptTemplate(template=template1)
             chain = LLMChain(llm=llm, prompt=prompt_template)
     # 运行链
             output = chain.run(input_params)
             if is_show:
                         print('生成的因子为:{}'.format(output))
             try:
                      data_list = json.loads(output)
                      return data_list
             except:
                 print('-------------- -因子改进失败，正在尝试重新生成--------')
                 continue
             
def select_random_dicts(dict_list:List[dict], n:int):
    """
    从一个包含字典的列表中随机选出 n 个字典。

    参数:
    dict_list (list): 包含字典的列表。
    n (int): 要选出的字典数量。

    返回:
    list: 随机选出的 n 个字典组成的新列表。
    """
    # 确保 n 不超过 dict_list 的长度
    if n > len(dict_list):
        raise ValueError("n 不能大于列表中的字典数量")

    # 从列表中随机选择 n 个字典
    selected_dicts = random.sample(dict_list, n)

    return selected_dicts


def filter_factor_info(factor_list:List[dict]):
    '''
    过滤因子信息，只保留Factor Formula,Explanation,rankIC_mean,ICIR
    
    '''
    # 定义要保留的键
    keys_to_keep = ['Factor Formula', 'Explanation', 'rankIC_mean','ICIR']

    # 创建一个新的列表，只包含所需的键值对
    filtered_list = [{k: factor[k] for k in keys_to_keep} for factor in factor_list]

    return filtered_list

def factor_loader(file_path:str='factor_kk.json',by:str='rankIC_mean',sort_opt:int=2,a:int=40,b:int=20):
     '''
     这个函数用于产生因子例子
     a:最好的几个
     b:从a中选几个

     返回包含Factor Formula,Explanation,rankIC_mean的list
     '''
     
     all_dict=show(file_path=file_path,by=by,sort_opt=sort_opt).head(a).to_dict(orient='records')
     select_random_dict=select_random_dicts(all_dict,b)
     filtered_list=filter_factor_info(select_random_dict)
     return filtered_list




    




