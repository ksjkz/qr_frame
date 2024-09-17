import json
import os
import pandas as pd

def save_dict_to_json(new_dict: dict, file_path: str = 'factor.jsonl', is_show=True):
    '''
    save_dict_to_json
    将输入的dict保存到指定 JSON Lines 文件
    每个字典都会作为一行追加到文件中
    '''
    # 使用追加模式 ('a') 写入文件，每次写入一行 JSON 对象
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(new_dict, file, ensure_ascii=False)
        file.write('\n')  # 每个 JSON 对象占一行
    
    if is_show:
        print(f'-----------------已将字典保存到 {file_path} ---------------')

'''
example_dict = {"name": "example", "value": 123}
save_dict_to_json(example_dict)
'''

def read_jsonl(file_path: str):
    '''
    逐行读取 JSON Lines 文件
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)

def jsonl_to_df(file_path: str) -> pd.DataFrame:
    '''
    读取 JSON Lines 文件并转换为 pandas DataFrame
    '''
    # 使用列表生成器将所有 JSON 对象收集到一个列表中
    data = list(read_jsonl(file_path))
    
    # 转换为 pandas DataFrame
    df = pd.DataFrame(data)
    return df