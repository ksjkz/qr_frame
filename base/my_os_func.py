import re
import os
from time import time as time_now
from datetime import datetime, date
from datetime import datetime, time
import pickle
import numpy as np
import pandas as pd
from functools import wraps

def get_all_dir(root_dir:str):
    """
    返回根目录下所有文件的路径 返回路径list
    """
    a=[]
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            a.append(file_path)
    return a


def filter_paths_by_suffix(paths:list, suffix:str=''):
    """
    筛选以什么什么为结尾的新列表
    比如以a.csv结尾的所有地址
    """
    filtered_paths = [path for path in paths if path.endswith(suffix)]
    return filtered_paths


def read_csv_from_paths_to_one_df(paths:list[str]):
    """
    根据地址读取多个csv,并将其存入一个df
    """
    dataframes = []
    for path in paths:
        if path.endswith('.csv') and os.path.isfile(path):
            df = pd.read_csv(path)
            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df



def save_csv_to_dir(input_file_path, output_directory):
    """
    读取 CSV 文件并将其保存到指定目录下。
    参数:
    - input_file_path: str, 输入 CSV 文件的路径 (file) like 'C:/2478/generic_bar\\2024\\07\\01\\ashare_lag_close_1day.csv'
    - output_directory: str, 保存文件的目标目录  (dir)  liek 'C:/2478

    返回:
    - None
    """
    try:
        # 确保目标目录存在
        os.makedirs(output_directory, exist_ok=True)
        df = pd.read_csv(input_file_path)
        # 构造输出文件路径
        output_file_path = os.path.join(output_directory, os.path.basename(input_file_path))
        df.to_csv(output_file_path, index=False)
        print(f"文件已成功保存到 {output_file_path}")
    except Exception as e:
        print(f"处理文件 {input_file_path} 时出错: {e}")


def get_last_n_dir(path:str,n:int=3):
    """
    获取路径的最后几级目录。

    参数:
    - path: str, 输入的文件或目录路径
    -  n :int,保留最后几集目录
    返回:
    - str: 最后三级目录的路径
    """
    # 去掉末尾的文件名或斜杠以确保路径只包含目录
    path = os.path.normpath(path)  # 标准化路径以处理不同操作系统的路径分隔符
    # 分割路径
    parts = path.split(os.sep)  # 使用操作系统的路径分隔符进行分割
    # 获取最后三级目录
    last__dirs = os.sep.join(parts[-n:-1])  # 连接最后三个部分
    return os.path.normpath(last__dirs)



    





    







    
















