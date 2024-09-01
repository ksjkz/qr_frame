'''
written by KSJKZ (QT)
git@github.com:ksjkz/qr_frame.git
'''
'''
数据集要求注释:
时序操作通过大量pd.rolling来实现. rolling是本行以及向上的n行.因此,数据集请处理为
pd.dataframe

ticker1   20231231
ticker1   20240101
ticker1   20240102
          ....

ticker2   20231231
ticker2   20240101
ticker2   20240102
          ....
的形式

如果只有一只ticker(或者其他类似ticker的东西)
则只需要设置时间为由早到晚


'''


import ast
import math
from typing import List, Dict,Tuple
import warnings
import pandas as pd
import numpy as np
import re
import numbers
from tqdm import tqdm
'''
ast 抽象语法树
python中的ast模块提供了方法,用于解析操作 python源代码(形式)的str变成抽象语法树

E.g.
import ast
expression = "div(ts_std(sub(ADJ_D_HIGH, ADJ_D_LOW), 10), ts_mean(sub(ADJ_D_HIGH, ADJ_D_LOW), 10)) * sqrt(YUAN_VOLUME)"
# 解析表达式生成AST
parsed_expr = ast.parse(expression, mode='eval')  
其中:        
'exec'：解析一个 Python 模块或脚本，适用于多个语句。
'eval'：解析一个单一表达式。
'single'：解析单个交互式语句，通常用于 REPL 环境


输出(ast.Expression对象)
Expression(
    body=BinOp(
        left=Call(
            func=Name(id='div', ctx=Load()), 
            args=[
                Call(
                    func=Name(id='ts_std', ctx=Load()),
                    args=[
                        Call(
                            func=Name(id='sub', ctx=Load()),
                            args=[
                                Name(id='ADJ_D_HIGH', ctx=Load()),
                                Name(id='ADJ_D_LOW', ctx=Load())],
                            keywords=[]),
                        Constant(value=10)],
                    keywords=[]),
                Call(
                    func=Name(id='ts_mean', ctx=Load()),
                    args=[
                        Call(
                            func=Name(id='sub', ctx=Load()),
                            args=[
                                Name(id='ADJ_D_HIGH', ctx=Load()),
                                Name(id='ADJ_D_LOW', ctx=Load())],
                            keywords=[]),
                        Constant(value=10)],
                    keywords=[])],
            keywords=[]),
        op=Mult(),
        right=Call(
            func=Name(id='sqrt', ctx=Load()),
            args=[
                Name(id='YUAN_VOLUME', ctx=Load())],
            keywords=[])))

在 Python 的抽象语法树（AST）中，Call 节点表示一个函数调用。Call 节点有多个子节点，描述了被调用的函数、传递的参数以及关键字参数等。

Call 节点的属性
以下是 Call 节点的主要属性：

func：表示被调用的函数。
args：表示传递给函数的参数列表。
keywords：表示关键字参数列表。           


ctx 属性用于指定 AST 节点的上下文类型，表示变量或表达式在代码中的角色。常见的上下文类型包括：

Load：变量或表达式正在被读取。
Store：变量或表达式正在被赋值。
Del：变量或表达式正在被删除。
AugLoad 和 AugStore：增强赋值操作中的读取和存储。
Param：函数参数。

'''

class Node:
    '''
    定义节点
    '''
    def __init__(self, value):
        self.value = value
        self.children = []


class ExpressionTree:
    def __init__(self, expression,columns_list):
        self.expression = expression
        self.columns_list=columns_list
        self.root = self._parse_expression(expression)

    def _parse_expression(self, expression):
        '''
        将expression解析为抽象语法树
        返回body,从而获取根节点
        根节点通过_ast_to_tree存储了全部内容
        '''
        try:
            ast_tree = ast.parse(expression, mode='eval')
        except Exception as e:
            print(f"Invalid expression: {expression}")
            return False
        return self._ast_to_tree(ast_tree.body)  #在上面的例子中body=BinOp() 返回的是BinOp对象

    def _ast_to_tree(self, node): 
        '''
        通过递归向节点里添加子节点
        '''
        if isinstance(node, ast.Call):
            root = Node(node.func.id)
            for arg in node.args:
                root.children.append(self._ast_to_tree(arg))
        elif isinstance(node, ast.BinOp):
            root = Node(type(node.op).__name__)
            root.children.append(self._ast_to_tree(node.left))
            root.children.append(self._ast_to_tree(node.right))
        elif isinstance(node, ast.Name):
            root = Node(node.id)
        elif isinstance(node, ast.Constant):
            root = Node(node.n)
        else:
            raise TypeError(f"Unsupported type: {type(node)}")
        return root

    def _preorder_traversal(self, node): 
        '''
        前序遍历
        '''   
        if node is None:
            return []
        result = [node.value]
        for child in node.children:
            result.extend(self._preorder_traversal(child))
        return result

    def get_preorder_expression(self):
        '''
        获取从根节点开始的前序表达式
        '''
        return self._preorder_traversal(self.root)
    
    def get_reconstruct_expression(self):
        '''
        由前序表达式获取重构的公式
        '''
        return reconstruct_expression(self.get_preorder_expression())

    
    def get_time_series_lenth(self,print_opt=False):
        '''
        获取时间序列长度
        '''
        return compute_time_series_length(self.root,print_opt=print_opt,columnss_list=self.columns_list)
    
'''
# 示例表达式
expression = "div(ts_std(sub(ADJ_D_HIGH, ADJ_D_LOW), 10), ts_mean(sub(ADJ_D_HIGH, ADJ_D_LOW), 10))*sqrt(YUAN_VOLUME)"
# 创建表达式树
expr_tree = ExpressionTree(expression)
# 获取前序表达式表示
preorder_expr = expr_tree.get_preorder_expression()
print("前序表达式表示:", preorder_expr)

结果:
<ast.BinOp object at 0x000001899B0AF8B0>
前序表达式表示: ['Mult', 'div', 'ts_std', 'sub', 'ADJ_D_HIGH', 'ADJ_D_LOW', 10, 'ts_mean', 'sub', 'ADJ_D_HIGH', 'ADJ_D_LOW', 10, 'sqrt', 'YUAN_VOLUME']
'''

arity = {
    # Element-wise Operators
    'add': 2,
    '*':2,
    'Add':2,

    'sub': 2,
    '-':2,
    'Sub':2,

    'mul': 2,
    '*':2,
    'Mult':2,

    'div': 2,
    '/':2,
    'Div':2,

    'pow':2,
    'sqrt': 1,
    'log': 1,
    'abs': 1,
    'neg': 1,
    'inv': 1,
    'max': 2,
    'min': 2,
    'sin': 1,
    'cos': 1,
    'tan': 1,
    'sig': 1,
    'sign': 1,
    'ltp': 2,
    'gtp': 2,

    # Time Series Operators
    'ts_sum': 2,
    'ts_prod': 2,
    'ts_covariance': 3,
    'ts_corr':3,

    'ts_std': 2,
    'ts_mean': 2,
    'ts_timeweighted_mean': 2,
    'ts_rank': 2,
    'ts_max': 2,
    'ts_min': 2,
    'ts_median':2,
    'ts_argmax': 2,
    'ts_argmin': 2,
    'ts_skew': 2, #计算偏度
}


columns_list=['D_OPEN_CHANGE', 'D_HIGH_CHANGE', 'D_LOW_CHANGE', 'D_CLOSE_CHANGE','D_VOLUME_CHANGE', 'D_PB_CHANGE', 'LOG_MCAP', 'CLASS_NAME', 'TICKER',
       'T_DATE', 'ONE_DAY_RETURN','ADJ_D_OPEN', 'ADJ_D_HIGH', 'ADJ_D_LOW', 'ADJ_D_CLOSE','MCAP_IN_CIRCULATION', 'YUAN_VOLUME', 'PB_RATIO', 'ONE_DAY_RETURN_PREV',] #这个是默认的columns_list,如果column不是这些,可将参数传入

def reconstruct_expression(preorder_expr:list)->str:
    '''
    输入前序表达式通过堆栈返回原始公式
    E.g. 
    输入:['Mult', 'div', 'ts_std', 'sub', 'ADJ_D_HIGH', 'ADJ_D_LOW', 10, 'ts_mean', 'sub', 'ADJ_D_HIGH', 'ADJ_D_LOW', 10, 'sqrt', 'YUAN_VOLUME']
    输出:'Mult(div(ts_std(sub(ADJ_D_HIGH, ADJ_D_LOW), 10), ts_mean(sub(ADJ_D_HIGH, ADJ_D_LOW), 10)), sqrt(YUAN_VOLUME))'
    '''
    stack = []
    # 操作数数量字典，假设我们知道每个函数或运算符的参数数量
    
    for token in reversed(preorder_expr): #翻转,从右往左处理
        if token in arity:
            args = [stack.pop() for _ in range(arity[token])]
            expr = f"{token}({', '.join(args)})"
            stack.append(expr)
        else:
            stack.append(str(token))
    return stack[0]


def preorder2DFoder(preorder_expr:list,std_opt=True,zero_opt=True,columns_list:list=columns_list)-> Tuple[List[str], int]:
    '''
    通过前序表达式通过堆栈返回df命令和命令步骤数(不算每步之后的标准化)
    std_opt:是否每一步计算后标准化
    zero_opt:是否将零值处理为0,0001,方便除法
    columns_list:公式中出现的列名
   

    E.g. 
    输入:['Mult', 'div', 'ts_std', 'sub', 'ADJ_D_HIGH', 'ADJ_D_LOW', 10, 'ts_mean', 'sub', 'ADJ_D_HIGH', 'ADJ_D_LOW', 10, 'sqrt', 'YUAN_VOLUME']
    输出:
    ["df['step_1']=np.sqrt(df['YUAN_VOLUME'])",
    "df['step_2']=df['ADJ_D_HIGH']-df['ADJ_D_LOW']",
    "df['step_3']=df['step_2'].rolling(window=10).mean()",
    "df['step_4']=df['ADJ_D_HIGH']-df['ADJ_D_LOW']",
    "df['step_5']=df['step_4'].rolling(window=10).std()",
    "df['step_6']=df['step_5']/df['step_3']",
    "df['step_7']=df['step_6']*df['step_1']"]
    
    '''
    stack = []
    index:int =0
    # 操作数数量字典，假设我们知道每个函数或运算符的参数数量
    df_oders=[]
    for token in reversed(preorder_expr): #翻转,从右往左处理
        if token in arity:
            index=index+1
            args = [stack.pop() for _ in range(arity[token])]
            match token:
                case 'add':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']+df['{args[1]}']")
                case '+':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']+df['{args[1]}']")
                case 'Add':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']+df['{args[1]}']")



                case 'sub':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']-df['{args[1]}']")
                case '-':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']-df['{args[1]}']")
                case 'Sub':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']-df['{args[1]}']")


                case 'mul':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']*df['{args[1]}']")
                case '*':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']*df['{args[1]}']")
                case 'Mult':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']*df['{args[1]}']")


                case 'div':
                    df_oders.append(f"df['step_{index}']=np.where(df['{args[1]}'] != 0, df['{args[0]}'] / df['{args[1]}'], np.nan)")
                case '/':
                    df_oders.append(f"df['step_{index}']=np.where(df['{args[1]}'] != 0, df['{args[0]}'] / df['{args[1]}'], np.nan)")
                case 'Div':
                    df_oders.append(f"df['step_{index}']=np.where(df['{args[1]}'] != 0, df['{args[0]}'] / df['{args[1]}'], np.nan)")


                case 'sqrt':
                    df_oders.append(f"df['step_{index}']=np.sqrt(df['{args[0]}'])")
                case 'pow':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}']**{args[1]}")
                case 'log':
                    df_oders.append(f"df['step_{index}']=np.log(df['{args[0]}'])")
                case 'abs':
                    df_oders.append(f"df['step_{index}']=np.abs(df['{args[0]}'])")
                case 'neg':
                    df_oders.append(f"df['step_{index}']=-df['{args[0]}']")
                case 'inv':
                    df_oders.append(f"df['step_{index}']=1/df['{args[0]}']")
                case 'max':
                    df_oders.append(f"df['step_{index}']=np.maximum(df['{args[0]}'],df['{args[1]}'])")
                case 'min':
                    df_oders.append(f"df['step_{index}']=np.minimum(df['{args[0]}'],df['{args[1]}'])")
                case 'sin':
                    df_oders.append(f"df['step_{index}']=np.sin(df['{args[0]}'])")
                case 'cos':
                    df_oders.append(f"df['step_{index}']=np.cos(df['{args[0]}'])")
                case 'tan':
                    df_oders.append(f"df['step_{index}']=np.tan(df['{args[0]}'])")
                case 'sig':
                    df_oders.append(f"df['step_{index}']=1/(1 + np.exp(df['{args[0]}']))")
                case 'sign':
                    df_oders.append(f"df['step_{index}']=np.sign(df['{args[0]}'])")
                case 'ltp':
                    df_oders.append(f"df['step_{index}']=df.apply(lambda row: 1 if row['{args[0]}'] > row['{args[1]}'] else 0, axis=1)")
                case 'gtp':
                    df_oders.append(f"df['step_{index}']=df.apply(lambda row: 1 if row['{args[0]}'] < row['{args[1]}'] else 0, axis=1)")
                case 'ts_sum':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).sum()")
                case 'ts_prod':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).apply(lambda x: x.prod(), raw=True)")
                case 'ts_covariance':
                    df_oders.append(f"df['step_{index}'] = df[['{args[1]}', '{args[0]}']].rolling(window={args[2]}).cov().iloc[0::2, -1].reset_index(drop=True)")   #cov出来是矩阵
                case 'ts_corr':
                    df_oders.append(f"df['step_{index}'] = df[['{args[1]}', '{args[0]}']].rolling(window={args[2]}).corr().iloc[0::2, -1].reset_index(drop=True)")  #corr出来的是矩阵
                case 'ts_mean':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).mean()")
                case 'ts_std':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).std()")
                case 'ts_timeweighted_mean':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).apply(lambda x: np.dot(x, np.arange(len(x), 0, -1)) / np.arange(len(x), 0, -1).sum(), raw=True)")
                case 'ts_rank':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).apply(lambda x: np.argsort(list(x)).argsort()[-1],raw=True)")
                case 'ts_max':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).max()")
                case 'ts_min':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).min()")
                case 'ts_median':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).median()")
                case 'ts_argmax':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).apply(lambda x: np.argmax(x),raw=True)")
                case 'ts_argmin':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).apply(lambda x: np.argmin(x),raw=True)")
                case 'ts_skew':
                    df_oders.append(f"df['step_{index}']=df['{args[0]}'].rolling(window={args[1]}).skew()")

            if std_opt:
                df_oders.append(f"df['step_{index}'] = (df['step_{index}'] - df['step_{index}'].mean()) / df['step_{index}'].std()")
            if zero_opt:
                df_oders.append(f"df['step_{index}'] = df['step_{index}'].apply(lambda x: x if abs(x) > 0.0001 else 0.0001)")

            expr = f"step_{index}"
            stack.append(expr)

        elif (token in columns_list) or isinstance(token, numbers.Number):
            stack.append(str(token))
        else:
            print(f"Invalid token: {token}")
            return False,False
    return df_oders,index


def find_Max_Number_In_Formulate(input_str:str)->int:
          '''
          这个函数用正则表达式找到公式中最大的数据,后面df计算时候每个ticker要舍弃前这么多行
          '''
          numbers = re.findall(r'\d+', input_str)
          numbers = [int(num) for num in numbers]
          max_number = max(numbers) if numbers else int(0)
          return max_number


def compute_time_series_length(node, cumulative_length=0,print_opt=False, columnss_list:list=columns_list):
    """
    输入 node
    返回值 公式时序长度-1
    即dfgroupby后前多少列为naN
   
    """
    if isinstance(node.value, (str, bytes)) and re.match(r"^ts_", node.value):#处理时序操作符
        if print_opt:
            print(node.value)
       
        lenth_list=[]
        for i in node.children:
            if isinstance(i.value, int):
               ts_length= i.value
               if print_opt:
                   print(ts_length)
               pass
            else:
                pre_length=compute_time_series_length(i, cumulative_length, print_opt=print_opt, columnss_list=columnss_list)
                lenth_list.append(pre_length)
        return  max(lenth_list)+ ts_length-1
    # 处理末端节点(列名orconstant)
    elif (node.value in columnss_list) or isinstance(node.value, numbers.Number):
        if print_opt:
            print(node.value)
        # 无时序影响，返回当前的累计长度
        return cumulative_length
    
    else:#处理无时序多元操作符
        if print_opt:
           print(node.value)
        lenth_list=[]
        # 递归计算左右子节点的时序长度，并取最大值
        for i in node.children:
           ts_length = compute_time_series_length(i, cumulative_length, print_opt=print_opt, columnss_list=columnss_list)
           lenth_list.append(ts_length)
        return max(lenth_list)
    



def f_coding(input:dict|str,df:pd.DataFrame,drop_opt=True,std_opt=False,zero_opt=True,columns_list:list=[],group_info=(0,'Ticker'))-> str|bool:
    '''
    输入dict形如:
    "Factor Name": "AA_2_price_volatility_and_trading_activity",
    "Factor Formula": "div(ts_std(sub(ADJ_D_HIGH, ADJ_D_LOW), 10), ts_timeweighted_mean(ADJ_D_CLOSE, 10)) * sqrt(YUAN_VOLUME) + div(ts_max(ADJ_D_HIGH, 10), ts_min(ADJ_D_LOW, 10)) * log(YUAN_VOLUME)"

    输入str 即公式本身 建议输入上面格式的dict

    输出
    dict的Factor Name 同时也是计算完成df的列名
    如果返回False说明有错误(这个函数不会raise)

    参数
    drop_opt是否要删除中间计算步骤的df列(step1...不包含最后的因子列)
    std_opt是否要标准化
    zero_opt是否要处理0值 max(接近于0,0.0001)
    columns_list: 需要被识别的列名,如果不传入会自动获取df的列名
    group_info: (is_group 1为要group,如果要group,groupby的列名值)

    注意预处理阶段就要把每个group交界的地方给扣掉
    '''
  
    match input:
        case dict():
            print("输入是一个字典 (dict)")
            try:
                input1 = input['Factor Formula']
                name_new = input['Factor Name']
            except KeyError as e:
                print(f"--------字典中缺少必要的键: {e}--------")
                return False
        
        case str():
            print("输入是一个字符串 (str)")
            input1 = input
            name_new = 'factor'
        
        case _:
            print('----------输入类型错误,必须是str或者dict,dict列名为"Factor Name""Factor Formula"--------')

            return False
        
    if columns_list==[]:
        columns_list=df.columns.to_list()
    
    print(f"df包含的列有{columns_list}")
   
    expr_tree = ExpressionTree(expression=input1, columns_list=columns_list)
    if expr_tree==False:
        print('----------------表达式(公式)不符合抽象语法树规范-------------')
        return False
    
    

    print(f"要解析的公式为\n{input1}")
    preorder_expr = expr_tree.get_preorder_expression()
    print('解析生成的前序表达式为\n{}'.format(preorder_expr))

    num_list=[item for item in preorder_expr if not isinstance(item, str)]
    for num in num_list:
        df[str(num)] = num
        print(f"生成df['{num}'] = {num} 列(为了处理非列名常数项,有可能这一列不参与计算)")
    
    df_orders,count=preorder2DFoder(preorder_expr,std_opt=std_opt,zero_opt=zero_opt,columns_list=columns_list)
    if  df_orders== False:
        print('----------------抽象语法树解析生成代码失败-------------')
        return False
    print('\n-------------每一步的代码已由抽象语法树解析生成---------------')
    for i in  df_orders:
        print(i)
    step_num=count   #步骤数
    steps_list = [f'step_{i}' for i in range(1, step_num + 1)] #[step_1, step_2,....]



    warnings.filterwarnings("ignore", category=FutureWarning)
    drop_column_list=steps_list
    df[drop_column_list]=0
    try:
        for code in tqdm( df_orders,leave=False,desc='正在计算因子值'):
               print('正在执行的代码是{}'.format(code))
               exec(code)
               print('----------------------此步骤执行完成---------------------------')
    except Exception as e:
               print(f"在执行解析生成的代码出现了错误: {e}")
               return False
    

    print('\n------------------全部计算步骤已经完成------------')
    if drop_opt:
        df.drop(columns=drop_column_list[:-1],inplace=True)
        df.drop(columns=[str(item) for item in num_list],inplace=True)

    print('\n--------------计算中出现的中间列已经delete------------')
    
    
    name_old=drop_column_list[-1]

    df.rename(columns={name_old: name_new}, inplace=True)
    print(f'\n---------------df生成新的一列并重新命名为{name_new}-----------')

    if group_info[0]==1:
        mm=expr_tree.get_time_series_lenth()
        print(f'------------------公式的时间序列长度为{mm}------------------')
        if mm !=0:
             indices = df.groupby(group_info[1]).apply(lambda x: x.head(mm).index).explode().values
             df.loc[indices,name_new]=None
             print(f'----------已经将每个{group_info[1]}的前{mm}个值设为空值--------')

    print('\n#########################因子值计算完成#################################')

    
    return name_new





