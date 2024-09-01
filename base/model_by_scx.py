import pandas as pd  
import numpy as np  
from sklearn.preprocessing import PolynomialFeatures  
import statsmodels.api as sm
from scipy.stats import spearmanr
import seaborn as sns  


pd.set_option('display.max_columns', None)


#df要求包含数据列，收益率列，日期列和股票代码列
def model(df, n, features, y_var='ONE_DAY_RETURN_PREV', degree=2, select=3):
    n_rows = df.shape[0]
    #数据预处理
    for feature in features:
        # mean = df.iloc[:, i].mean()
        # std = df.iloc[:, i].std()
        # df.iloc[:, i] = (df.iloc[:, i] - mean) / std#标准化
        # print(df.groupby(df['T_DATE'], as_index=False, group_keys=False, sort=False)[feature].apply(lambda x: (x - x.mean()) / x.std()))
        df.loc[:, feature] = df.groupby(df['T_DATE'], as_index=False, group_keys=False, sort=False)[feature].apply(lambda x: (x - x.mean()) / x.std())
    df.loc[:, y_var] = df.groupby('T_DATE', as_index=False, group_keys=False, sort=False)[y_var].apply(lambda x: x - x.mean())
    print(f"df shape after standardization {df.shape[0] / n_rows:.1%}")
    df = df.loc[df[y_var].abs() <= 0.21]#删除极端不合理数值
    print(f"df shape after <= 0.21 {df.shape[0]/n_rows:.1%}")
    df = df.dropna(thresh=n)  #删除全为nan的行
    print(f"df shape after dropping all nan {df.shape[0]/n_rows:.1%}")
    df = df.fillna(0) #为剩下的nan值赋0

    x_vars = np.random.choice(features, size=select, replace=False)#随机选取select列作为拟合函数的数据
    print(f"x_vars: {x_vars}")

    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)  
    X = df[x_vars].values
    if all(col in df.columns for col in x_vars):  
        print("所有列名都存在")  
    else:  
        print("存在不存在的列名")   
  
    # 检查 X 是否为空  
    if X.shape[0] == 0:  
        print("没有样本数据")  
    else:  
        print("样本数据已正确提取")    
    X_poly = poly.fit_transform(X)  
    X_poly_with_intercept = sm.add_constant(X_poly) # 添加常数项以适合截距    
    # X_poly_with_intercept = X_poly
    model = sm.OLS(df[y_var], X_poly_with_intercept).fit() # 使用OLS进行回归    
    predictions = model.predict(sm.add_constant(poly.transform(df[x_vars].values))) # 使用模型进行预测，注意这里也添加常数项    
    df['Prediction'] = predictions # 将预测结果作为新列添加到原始DataFrame中 
    return df  




def test(df):
    df['ret'] = np.log(df['ONE_DAY_RETURN_PREV'] + 1)
    df['ret_demean'] = df.groupby(by='T_DATE', as_index=False, group_keys=False)['ONE_DAY_RETURN_PREV'].apply(lambda x: x - np.nanmean(x))
 # 确保必要的列存在  
    required_columns = ['T_DATE', 'ONE_DAY_RETURN_PREV', 'Prediction', 'ret_demean']  
    if not all(col in df.columns for col in required_columns):  
        raise ValueError("Missing required columns in DataFrame")  
  
    # 计算信息系数（IC）  
    rank_ic = df.groupby('T_DATE').apply(lambda x: spearmanr(x['ret_demean'], x['Prediction'])[0])  
    print("IC Statistics:")  
    print(rank_ic.describe())  
  
    # 计算IC的滚动平均值  
    rank_ic_ma = rank_ic.rolling(21).mean()  
    sns.lineplot(data=rank_ic_ma)  
      # 确保图表显示出来  
  
    # 使用分位数切割对预测值进行分组  
    df['q_cut'] = pd.qcut(df['Prediction'], q=10, labels=[f"q_{i}" for i in range(10)])  
  
    # 计算分组内去均值后的收益率的平均值及其累积收益率  
    df_q_cut_ret = df.groupby(['T_DATE', 'q_cut'])['ret_demean'].mean().unstack()  
    # df_q_cut_cumret = (df_q_cut_ret + 1).cumprod()
    df_q_cut_cumret = df_q_cut_ret.cumsum()
    print(df_q_cut_ret.describe(), df_q_cut_cumret.describe())
    df_q_cut_cumret.index = pd.to_datetime(df_q_cut_cumret.index)
    sns.lineplot(df_q_cut_cumret.iloc[:, :])
  
  
    # 显示统计信息  
    print("Grouped Return Statistics:")  
    print(df_q_cut_ret.describe())  
    print("Cumulative Return Statistics:")  
    print(df_q_cut_cumret.describe())  
  
    # 将索引转换为日期类型并绘制累积收益率



