
from langchain.memory import ConversationBufferMemory
import yaml
from langchain_openai import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import ConversationChain
from typing import List, Dict
import pandas as pd 
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import json
import os
import math
from IPython.display import clear_output
from base.factor_evaluate import Eval
import time
import random
from base.formulate_coding_AST import *

memory=ConversationBufferMemory()

def open_New_LLM():
     '''
open_New_LLM
用于创建LLm,这个函数会按照配置文件自动生成一个LLM
'''
     with open("config/llm_api.yaml", "r", encoding="utf-8") as file:
          config1 = yaml.safe_load(file)
     llm = ChatOpenAI(
          openai_api_base=config1['openai_api']['url'],
          openai_api_key=config1['openai_api']['key'],
          model = config1['openai_api']['model'],
          temperature=config1['openai_api']['temperature']
          )
     return llm

def test_llm_connection() -> bool:
    '''
test_llm_connection
用于测试LLM连接是否正常
'''
    try:
        # 初始化 OpenAI LLM
        llm = open_New_LLM()
        # 创建对话链
        conversation = ConversationChain(llm=llm)
        # 进行一次简单的对话
        response = conversation.run("Hello, how are you today?")
        # 检查响应是否为空或无效
        if response:
            print('LLM连接成功')
            return True
        else:
            print('LLM连接失败')
            return False
    except Exception as e:
        # 捕获任何异常并返回 False
        print(f"Connection failed: {e}")
        return False

def read_json_file(filename:str='factor_kk.json'):
    '''
    read_json_file
    读取指定的 JSON 文件并返回解析后的数据。
    参数:
    filename (str): JSON 文件的路径。
    返回:
    dict or list: 解析后的 JSON 数据。
'''
   
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 {filename} 不存在。")
    except json.JSONDecodeError:
        print(f"文件 {filename} 不是有效的 JSON。")
    except Exception as e:
        print(f"读取文件 {filename} 时发生错误: {e}")

def clean_json_1(data:list[dict],threshold:float=0.5)->list:
    '''
    输入list[dict],筛选其中没有nan和rankIC_mean绝对值小于threshold的dict,保存到新list
    返回新的list
    
    '''
    cleaned_data = []
    for item in data:
        has_nan = any(math.isnan(value) if isinstance(value, float) else False for value in item.values())
        if not has_nan and abs(item.get("rankIC_mean", 0)) <= threshold:
            cleaned_data.append(item)
    return cleaned_data

def clean_json_2(data:list[dict],typical_factor:str='ONE_DAY_RETURN')->list:
    '''
    输入list[dict],筛选其中没有typical_factor关键字的dict,保存到新list
    返回新的list
    
    '''
    cleaned_data = []
    for item in data:
        fomula=item.get('Factor Formula')
        try:
               A=ExpressionTree(fomula).get_preorder_expression()
               if not(typical_factor in A):
                       cleaned_data.append(item)
        except:
            
            continue
    return cleaned_data

def save_to_json(data, filename):
    '''
    保存list到指定json文件
    '''
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def clean_json(file_path:str='factor_kk.json',new_file_path:str='factor_kk_clean.json',threshold:float=0.5,typical_factor:str='ONE_DAY_RETURN'):
    '''
    输入json 清洗 保存到新json
    
    '''
    data=read_json_file(file_path)
    data=clean_json_1(data,threshold=threshold)
    data=clean_json_2(data,typical_factor=typical_factor)
    save_to_json(data, new_file_path)
    print('已经成功清洗json并保存到{}'.format(new_file_path))
    return True

def save_dict_to_json(new_dict: dict, file_path: str = 'factor.json',is_show=True):
    '''
save_dict_to_json
将输入的dict保存到指定json文件
注意不会覆盖原来文件中已有内容
'''
    file_exists = os.path.exists(file_path)
    
    if not file_exists:
        # 如果文件不存在，创建文件并写入 JSON 数组的开头
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('[\n')
            json.dump(new_dict, file, ensure_ascii=False, indent=4)
            file.write('\n]')
    else:
        # 如果文件存在，读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # 移除最后一个字符（应该是 `]`）
            content = content.strip()
            if content.endswith(']'):
                content = content[:-1]
            # 如果文件中已经有内容，则在末尾添加逗号
            if len(content) > 1:
                content += ',\n'

        # 追加新的字典并重新写入文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            json.dump(new_dict, file, ensure_ascii=False, indent=4)
            file.write('\n]')
    if is_show:
        print('-----------------已将文件保存到{}---------------'.format(file_path))

def show(file_path='dataset/all_factor_sorted.csv',key:str='df',return_opt:int=0,by:str='rankIC_mean',sort_opt:int=2,):
    '''
    输入地址(json or csv or h5 还要额外提供key) 按照by列排序,返回排序后的df
    by:按照怎么指标排序,默认是rankIC_mean
    return_opt:返回类型,0是返回df,1是返回list[dict]
    sort_opt:排序方式,0是升序,1是降序,2是绝对值降序,默认为2
   
    '''
    if re.match(r'.*\.json$', file_path, re.IGNORECASE) is not None:
         data=read_json_file(file_path)
         df = pd.DataFrame(data)
         df=df.dropna(subset=[by])
    elif re.match(r'.*\.csv$', file_path, re.IGNORECASE) is not None :
         df=pd.read_csv(file_path)
         df=df.dropna(subset=[by])
    elif re.match(r'.*\.h5$', file_path, re.IGNORECASE) is not None :
         df=pd.read_hdf(file_path,key=key)
         df=df.dropna(subset=[by])
    else:
         print(f'file_path must be a json csv or h5 file ,but get {file_path}')

    match sort_opt:
       case 0:
        # 按照by列升序排序
        df_sorted = df.sort_values(by=by, ascending=True)
       case 1:
        # 按照by列降序排序
        df_sorted = df.sort_values(by=by, ascending=False)
       case 2:
        # 按照绝对值降序排序
        df_sorted = df.iloc[df[by].abs().argsort()[::-1]]
       case _:
        raise ValueError("sort_opt must be 0, 1, or 2")
    df_sorted=df_sorted.drop_duplicates(subset=by, keep='first')
    if return_opt==0:
       return df_sorted
    elif return_opt==1:
       return df_sorted.to_dict('records')

def f_c_eval(df:pd.DataFrame,input:list[dict],save_opt=True,save_path:str='factor_change.json',period:str='',drop_opt1=True,drop_opt2=True,wait_opt=True):
    '''
    对输入的因子list进行因子值计算和回测,返回包含回测信息的因子list
    df:全局变量 包含因子计算需要信息的df
    input:因子list
    drop_opt1:是否删除df计算中出现的步骤
    drop_opt2:是否删除df中已经计算过的因子值这一列,防止df过大出现memory error
    wait_opt:是否执行完一轮后等待20s方便查看打印
    period:添加的时间标签 i[f'IC_mean{period}']=e.IC_mean
    '''
    aa=[]
    if isinstance(input, dict):
        input = [input]
    input_lenth=len(input)

    for index,i in enumerate(input):
              try:
                      print('总共{}个因子 现在计算第{}个因子  因子公式：{}'.format( input_lenth,index+1,i['Factor Formula']))
                     
                      name=f_coding(i,df=df,drop_opt=drop_opt1)
                      print(name)
                      if name==False:
                           print('coding失败')
                           clear_output(wait=True)
                           time.sleep(20)
                           continue
                      e=Eval(df=df,factor_name=name)
                      e.cul()
                      i[f'IC_mean{period}']=e.IC_mean
                      i[f'rankIC_mean{period}'] = e.rankIc_mean
                      i[f'ICIR{period}']=e.ICIR
                      i[f'autocorr{period}']=e.autocorr
                      if save_opt:
                           save_dict_to_json( new_dict=i,file_path=save_path,is_show=True )
                      if drop_opt2:
                           df.drop(columns=name,inplace=True) #防止df过大出现memory error
                      if wait_opt:
                            time.sleep(60)
                      clear_output(wait=True)
                      aa.append(i)

              except Exception as e:
                      print(e)
                      print("运算因子值出错开始等待...")
                      print('运算出错的因子是 因子公式：{}'.format(i['Factor Formula']))
                      time.sleep(30)  # 等待600秒（10分钟）
                      print("等待结束")
                      continue
    return aa
    
def set_complexity():
    complexities = [  'complex', 'very complex', 'extremely complex', 'extremely very complex','very complex','extremely complex']
    complexity1 = random.choice(complexities)
    return complexity1


