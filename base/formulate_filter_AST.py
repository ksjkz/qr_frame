from base.formulate_coding_AST import*
import pandas as pd

def mapping_str(strr: str):
    '''
    将前序表达式每个字符映射成数字(注意映射的数字不能出现0,因为0是一个标志)
    '''
    mapping = {
        'add': 11, '+': 11, 'Add': 11,
        'sub': 12, '-': 12, 'Sub': 12,
        'mul': 13, '*': 13, 'Mult': 13,
        'div': 14, '/': 14, 'Div': 14,
        'sqrt': 15, 'pow': 16, 'log': 17,
        'abs': 18, 'neg': 19, 'inv': 21,
        'max': 22, 'min': 23, 'sin': 24,
        'cos': 25, 'tan': 26, 'sig': 27,
        'sign': 28, 'ltp': 29, 'gtp': 31,
        'ts_sum': 32, 'ts_prod': 33, 'ts_covariance': 34,
        'ts_corr': 35, 'ts_mean': 36, 'ts_std': 37,
        'ts_timeweighted_mean': 38, 'ts_rank': 39, 'ts_max': 41,
        'ts_min': 42, 'ts_median': 43, 'ts_argmax': 44,
        'ts_argmin': 45, 'ts_skew': 46,
        'D_OPEN_CHANGE': 47, 'D_HIGH_CHANGE': 48, 'D_LOW_CHANGE': 49,
        'D_CLOSE_CHANGE': 51, 'D_VOLUME_CHANGE': 52, 'D_PB_CHANGE': 53,
        'LOG_MCAP': 54, 'ADJ_D_CLOSE': 55, 'ADJ_D_HIGH': 56,
        'ADJ_D_OPEN': 57, 'ADJ_D_LOW': 58, 'MCAP_IN_CIRCULATION':59,
        'YUAN_VOLUME': 61, 'PB_RATIO': 62, 'ONE_DAY_RETURN_PREV': 63,
    }
    return mapping.get(strr, 99)#常数返回99
          

def count_nonzero_digits(number):
    '''
    通过统计Factor Number里面非0数量来计算公式长度
    '''
    number_str = str(number)
    nonzero_count = sum(1 for char in number_str if char != '0')
    nonzero_count=nonzero_count/2
    return nonzero_count


def trans_preorder_to_scalar(fomulate:str):
    '''
    将公式映射到一个数值Factor Number
    可以通过
    df['Factor Number']=df['Factor Formula'].apply(trans_preorder_to_scalar)
    df['kk']=df['Factor Number']//10000
    df['kk']=df['kk']//10000
    df = df.drop_duplicates(subset='kk', keep='first')
    来筛选结构相似的公式
    '''
    A=ExpressionTree(fomulate)
    preorder=A.get_preorder_expression()
    sum=0
    for index,i in enumerate(preorder):
        num=mapping_str(i)
        sum=sum+num*(10**(-2*index+16))

    return sum



def filter(d1:pd.DataFrame,n:int=4,filter_opt:int=0):
    '''

    
    '''

    df=d1
    df['Factor Number']=df['Factor Formula'].apply(trans_preorder_to_scalar)
    df['Formula_lenth']=df['Factor Number'].apply(count_nonzero_digits)
    df['category']=df['Factor Number']//(10**(n*2))
    print('embedding,计算公式长度,分类完成 结果记录在,三列分别是Factor Number Formula_lenth category')
    if filter_opt==0:
        df = df.loc[df.groupby('category')['Formula_lenth'].apply(lambda x: x.idxmin())].reset_index(drop=True)
        print(f'取公式最短的来去重,还剩{df.shape[0]}行')
    elif filter_opt==1:
        df = df.drop_duplicates(subset='category', keep='first')
        print(f'取每组第一行来去重,还剩{df.shape[0]}行')

    return df

        

