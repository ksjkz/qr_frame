import pandas as pd
import re

Time_Boundary1=pd.Timestamp('2019-01-01')
Time_Boundary2=pd.Timestamp('2024-01-01')

class DataLoader:
    '''
    从dataset中加载数据到pd.dataframe中
    可输入文件格式 .csv .h5 默认为data.h5 如果是h5,需要额外标注数据的key是什么
    sorted_opt=False 相当于groupby 股票
    sorted_opt=True  相当于groupby 日期
    
    
    '''
    def __init__(self, file_path:str='data.h5',key:str='df',sorted_opt=False,):
        print(f"Loading data from {file_path}...")
        if re.match(r'.*\.h5$', file_path, re.IGNORECASE) is not None:
             df = pd.read_hdf(file_path, key=key)
             df=df.dropna()
             df['T_DATE'] = pd.to_datetime(df['T_DATE'])
      
        elif re.match(r'.*\.csv$', file_path, re.IGNORECASE) is not None :
       
             df=pd.read_csv(file_path)
             df=df.dropna()
             df.reset_index(inplace=True,drop=True)
             df.drop(columns=[ 'Unnamed: 0'], inplace=True)
             df['T_DATE'] = pd.to_datetime(df['T_DATE'])

        if sorted_opt:
            self.df=df.groupby('T_DATE').apply(lambda x: x).reset_index(drop=True)
            print('df是日期groupby')
            
        else:
            self.df=df
            print('df是按股票groupby')
            
        self.columns=df.columns
        print(self.columns)
        self.row_lenth=df.shape[0]
        print(self.row_lenth)
        

    def get_Trainset(self):
        d1=self.df.copy()
        print('df数据早于{}  '.format(Time_Boundary1))
        return d1[d1['T_DATE'] < Time_Boundary1]
    def get_Validateset(self):
        d3=self.df.copy()
        print('df数据开始于{}   结束于{}'.format(Time_Boundary1,Time_Boundary2))
        return d3[(d3['T_DATE'] >= Time_Boundary1) & (d3['T_DATE'] < Time_Boundary2)]
    def get_Testset(self):
        d2=self.df.copy()
        print('df数据晚于{}  '.format(Time_Boundary2))
        return d2[d2['T_DATE'] >= Time_Boundary2]
    
