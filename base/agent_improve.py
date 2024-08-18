import yaml
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict
from base.memory import *
from base.factor_generate import *
import random
'''
两个agent,一个分析因子好坏和提出修改意见
另一个修改生成新的因子


'''

def factor_loader2(file_path:str='factor_kk.json',by:str='rankIC_mean',sort_opt:int=2,split_num:int=20,split_thres:int=4,good_num:int=20,medium_num:int=4,bad_num:int=4):
    '''
    用于从保存因子的json文件中随机挑选一些因子,返回三个list,分别为好中坏的因子list
    file_path:因子json文件路径
    by:排序依据
    sort_opt:排序方式
    split_num:分成几份  默认为20组分
    split_thres:   分成3份,第一份是split_num的第一组分,第二份是split_num的第一组分到 split_thres组分,第三份是剩下的
    good_num:从第一份中选几个因子
    medium_num:从第二份中选几个因子
    bad_num:从第三份中选几个因子
    '''
    df=show(file_path=file_path,by=by,sort_opt=sort_opt)
    n=df.shape[0]
    split_size = n // split_num
    df1 = df.iloc[:split_size].to_dict(orient='records')
    df2 = df.iloc[split_size:split_thres*split_size].to_dict(orient='records')
    df3 = df.iloc[split_thres*split_size:].to_dict(orient='records')
    select_random_dict1=select_random_dicts(df1,good_num)
    select_random_dict2=select_random_dicts(df2,medium_num)
    select_random_dict3=select_random_dicts(df3,bad_num)
    filtered_list1=filter_factor_info(select_random_dict1)
    filtered_list2=filter_factor_info(select_random_dict2)
    filtered_list3=filter_factor_info(select_random_dict3)
    return filtered_list1, filtered_list2, filtered_list3

def select_top_n_factor(list1:list[dict], list2:list[dict], n:int=8,by:str='rankIC_mean'):
    '''
    输入两个因子list,返回绝对值rankic最高的前n-1个因子加一个后面的随机因子组成的新的list
    
    
    '''
    combined = list1 + list2
    sorted_combined = sorted(combined, key=lambda x: abs(x.get(by, 0)), reverse=True)
    top_n = sorted_combined[:n-1]+[sorted_combined[random.randint(n-1,len(sorted_combined)-1)]]
    print(f"选出的最好{n}个因子为:")
    for i in top_n:
         print(i)
    return top_n

class FactorGenerationAgent:
    '''
    因子生成agent
    根据anaylsis生成因子list
    flag:指示不同的template
    
    
    '''
    def __init__(self, llm=open_New_LLM(), verbose=True,flag=True):
        with open("config/params_for_agent_improve.yaml", "r", encoding="utf-8") as file:
          config= yaml.safe_load(file)
        if flag:
             input_params = config["input_params1"]
        else:
             input_params = config["input_params"]

        input_params['operators'] = config["operators"]
        
         
        self.input_params=input_params
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template="""
                你的工作是因子生成。因子用于股票投资,因子的rankIC_mean绝对值越高,说明因子预测能力越好。ICIR绝对值越高,说明因子越稳定
                根据因子分析员的分析和你的理解，在原来的因子上,生成新的因子：
                分析：{analysis}
                你可以使用的方法包括但不限于:
                       -  (Genetic Programming，GP) 
                       -  (Particle Swarm Optimization，PSO)
                       -  (Simulated Annealing，SA)
                       -  (Differential Evolution，DE)
                       -  (Ant Colony Optimization，ACO)
                       -  (Bee Algorithm，BA)
                       -  (Tabu Search，TS)
                       -  (Artificial Immune System，AIS)
                       -  (Shuffled Frog Leaping Algorithm，SFLA)
                       -  (Cultural Algorithm，CA)
                       -  (Co-Evolution)
   
                这个改进后的因子公式必须是{complexity}
                
                在因子公式中, {factor_terms} 必须被写作 {factor_term_names}. 有且只有这些变量
                公式包含这些运算元必须写作 {operators}  
                请直接输出格式为list[dict],包含{factor_num}个新因子
                dict中包含
                       "Factor Name": "随机数字_"+'_'+'base on its Explanation, not too long'  # like 90236563612381_CAPITALIZATION_ADJUSTED_RETURN
                       "Factor Formula": "The formula of the random factor"#公式必须写成python的语法格式,可以被抽象语法树解析
                       "Explanation": "A very very simple explanation of its meaning and how it is improved"
                 
                """
                
            ),
            verbose=verbose
        )
    
    def generate(self, analysis:str,factor_num:int=4,complexity:str='very complex',is_show=True)->list[dict]:
        self.input_params['factor_num']=factor_num
        self.input_params['complexity'] = complexity
        self.input_params['analysis'] = analysis

        output=self.chain.run(self.input_params)
        if is_show:
                         print('生成的因子为:{}'.format(output))
        start = output.find('[')
        end = output.rfind(']') + 1
        output = output[start:end]
             
        data_list = json.loads(output)

        return data_list

class FactorComparisonAgent:
    '''
    因子对比分析agent
    输入因子list
    str:anaysis,包含因子信息
    '''
    def __init__(self, llm=open_New_LLM(), verbose=True):
        self.chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template="""
                你的工作是结合给出的好的样本因子分析改进因子(因子用于股票投资,因子的rankIC_mean绝对值越高,说明因子预测能力越好。ICIR绝对值越高,说明因子越稳定)
                好的样本因子：{good_factors}
                需要被改进因子：{generated_factors}
                请结合好的样本因子的结构,背后的经济学含义等,结合你的知识库
                提出对需要被改进因子的改进建议(尽量精简)
                改进建议格式要求 原来因子式 该因子存在的问题 该如何改进(不要直接生成因子式)
                """
            ),
            verbose=verbose
        )
    
    def compare(self, generated_factors:list[dict], good_factors:list[dict])->str:
        return self.chain.run(
            good_factors=good_factors,
            generated_factors=generated_factors,
            
        )









     




